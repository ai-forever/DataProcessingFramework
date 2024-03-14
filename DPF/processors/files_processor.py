from typing import Any, Callable, Optional

import pandas as pd

from DPF.configs import FilesDatasetConfig
from DPF.connectors import Connector
from DPF.dataloaders import FilesDataset, identical_preprocess_function
from DPF.datatypes import ColumnDataType, FileDataType
from DPF.modalities import ModalityName
from DPF.types import ModalityToDataMapping
from DPF.validators import ValidationResult
from DPF.validators.format_validators import FilesValidator

from .processor import DatasetProcessor
from .processor_mixins import ApplyTransformProcessorMixin


class FilesDatasetProcessor(DatasetProcessor, ApplyTransformProcessorMixin):
    connector: Connector
    df: pd.DataFrame
    config: FilesDatasetConfig

    def __init__(
        self,
        connector: Connector,
        df: pd.DataFrame,
        config: FilesDatasetConfig,
    ):
        super().__init__(connector, df, config)

    def rename_columns(self, column_map: dict[str, str], workers: int = 1) -> list[str]:
        df = self.connector.read_dataframe(self.config.table_path)
        for col_old, col_new in column_map.items():
            assert col_old in df.columns, f'Dataframe dont have "{col_old}" column'
            assert col_new not in df.columns, f'Dataframe already have "{col_new}" column'
        df.rename(columns=column_map, inplace=True)
        self.connector.save_dataframe(df, self.config.table_path, index=False)

        self._df.rename(columns=column_map, inplace=True)
        return []

    def delete_columns(self, columns: list[str], workers: int = 1) -> list[str]:
        df = self.connector.read_dataframe(self.config.table_path)
        for col in columns:
            assert col in df.columns, f'Dataframe dont have "{col}" column'
        df.drop(columns=columns, inplace=True)
        self.connector.save_dataframe(df, self.config.table_path, index=False)

        self._df.drop(columns=columns, inplace=True)
        return []

    def update_columns(self, columns: list[str], workers: int = 1) -> list[str]:
        key_column = None
        for d in self.config.datatypes:
            if isinstance(d, FileDataType):
                key_column = d.modality.path_column
                break
        assert key_column is not None, "Cant find key column to use for update"
        assert key_column not in columns, f'Cant update key column "{key_column}"'

        df_old = self.connector.read_dataframe(self.config.table_path)
        df_new = self._df[[key_column] + columns]
        df_new.loc[:,key_column] = df_new[key_column].str.slice(len(self.config.base_path)+1)
        assert key_column in df_old.columns, f'Dataframe dont have "{key_column}" column'
        assert set(df_old[key_column]) == set(df_new[key_column]), \
            f'Dataframe has different values in "{key_column}"'

        duplicates = df_old[df_old[key_column].duplicated()][key_column].tolist()
        assert len(duplicates) == 0, f'Dataframe has duplicates in "{key_column}" column: {duplicates}'

        duplicates = df_new[df_new[key_column].duplicated()][key_column].tolist()
        assert len(duplicates) == 0, f'New dataframe has duplicates in "{key_column}" column: {duplicates}'

        assert len(df_old) == len(df_new), 'Length of dataframe is changed'

        columns_to_add = [i for i in df_new.all_columns if i != key_column]
        columns_intersection = set(df_old.columns).intersection(set(columns_to_add))

        if len(columns_intersection) > 0:
            df_old.drop(columns=list(columns_intersection), inplace=True)

        df = pd.merge(df_old, df_new, on=key_column)
        self.connector.save_dataframe(df, self.config.table_path, index=False)
        return []

    def validate(
        self,
        validate_filestructure: bool = True,
        validate_metadata: bool = True,
        columns_to_check: Optional[list[str]] = None,
        workers: int = 1,
        pbar: bool = True
    ) -> ValidationResult:
        if columns_to_check is None:
            columns_to_check = []
        validator = FilesValidator(
            self.df,
            self.connector,
            self.config,
            columns_to_check
        )
        return validator.validate(
            validate_filestructure=validate_filestructure,
            validate_metadata=validate_metadata,
            workers=workers,
            pbar=pbar
        )

    def _get_torch_dataset(
        self,
        modalities: list[ModalityName],
        columns_to_use: Optional[list[str]] = None,
        preprocess_f: Callable[[ModalityToDataMapping, dict[str, str]], Any] = identical_preprocess_function,
        return_none_on_error: bool = False
    ) -> FilesDataset:
        assert len(set(modalities)) == len(list(modalities))
        datatypes_to_load = [self.config.modality2datatype[m] for m in modalities]
        return FilesDataset(
            self.connector,
            self._df,
            datatypes_to_load,  # type: ignore
            metadata_columns=columns_to_use,
            preprocess_function=preprocess_f,
            return_none_on_error=return_none_on_error
        )

    def _read_sample_data(
        self,
        sample: dict[str, str]
    ) -> ModalityToDataMapping:
        path_column2modality: dict[str, ModalityName] = {}
        column2modality: dict[str, ModalityName] = {}
        for d in self.config.datatypes:
            if isinstance(d, ColumnDataType):
                column2modality[d.column_name] = d.modality.name
            elif isinstance(d, FileDataType):
                path_column2modality[d.modality.path_column] = d.modality.name
            else:
                raise ValueError()

        modality2data: ModalityToDataMapping = {}
        # read files
        for col in path_column2modality.keys():
            modality = path_column2modality[col]
            file_bytes = self.connector.read_file(sample[col], binary=True).getvalue()
            modality2data[modality] = file_bytes
        # read data from columns
        for col in column2modality.keys():
            modality = column2modality[col]
            modality2data[modality] = sample[col]
        return modality2data
