from functools import partial
from typing import Optional, Union

import pandas as pd
from tqdm.contrib.concurrent import process_map

from DPF.configs import (
    DatasetConfig,
    FilesDatasetConfig,
    ShardedFilesDatasetConfig,
    ShardsDatasetConfig,
)
from DPF.datatypes import FileDataType, ShardedDataType
from DPF.filesystems import FileSystem, LocalFileSystem
from DPF.processors import (
    DatasetProcessor,
    FilesDatasetProcessor,
    ShardedFilesDatasetProcessor,
    ShardsDatasetProcessor,
)

from .dataset_reader_utils import get_path_filename, read_and_validate_df


class DatasetReader:
    """Fabric for DPF.processors.DatasetProcessor

    Attributes
    ----------
    filesystem: FileSystem
        Filesystem to read datasets from
    """
    filesystem: FileSystem

    def __init__(self, filesystem: Optional[FileSystem] = None):
        """
        Parameters
        ----------
        filesystem: Optional[FileSystem] = None
            Instance of a filesystem to use. LocalFileSystem used by default
        """
        if filesystem is None:
            filesystem = LocalFileSystem()
        self.filesystem = filesystem

    def _read_and_validate_dataframes(
        self,
        datafiles: list[str],
        config: DatasetConfig,
        validate_columns: bool = True,
        processes: int = 1,
        progress_bar: bool = False,
    ) -> list[tuple[str, pd.DataFrame]]:
        if len(datafiles) == 0:
            raise ValueError("No datafiles in this path")

        required_columns = config.user_column_names if validate_columns else None

        worker_co = partial(read_and_validate_df, self.filesystem, required_columns)
        paths_dataframes: list[tuple[str, pd.DataFrame]] = process_map(
            worker_co, datafiles,
            max_workers=processes,
            chunksize=1,
            disable=not progress_bar
        )

        if validate_columns:
            self._validate_dataframes_columns(config, paths_dataframes)
        return paths_dataframes

    @staticmethod
    def _validate_dataframes_columns(
        config: DatasetConfig,
        paths_dataframes: list[tuple[str, pd.DataFrame]],
    ) -> None:
        required_columns = config.user_column_names

        column_set = set(paths_dataframes[0][1].columns.tolist())
        for path, df in paths_dataframes:
            df_columns = set(df.columns.tolist())

            for col in required_columns:
                assert col in df_columns, f'Expected {path} to have "{col}" column'
            assert df_columns == column_set, (
                f"Dataframe {path} have different columns. "
                f"Expected {column_set}, got {set(df.columns)}"
            )

    @staticmethod
    def _convert_sharded_columns_to_path_columns(
        split_suffix: str,
        config: Union[ShardedFilesDatasetConfig, ShardsDatasetConfig],
        df: pd.DataFrame
    ) -> pd.DataFrame:
        split_container_path = config.path.rstrip('/')+'/'+df['split_name']+split_suffix+'/'
        cols_to_drop = set()

        for datatype in config.datatypes:
            if isinstance(datatype, ShardedDataType):
                file_name_column = datatype.modality.sharded_file_name_column
                df[datatype.modality.path_column] = split_container_path+df[file_name_column]
                cols_to_drop.add(file_name_column)

        if len(cols_to_drop) > 0:
            df.drop(columns=list(cols_to_drop), inplace=True)
        return df

    @staticmethod
    def _rearrange_dataframe_columns(
        df: pd.DataFrame,
        config: DatasetConfig
    ) -> pd.DataFrame:
        column_index = []
        for datatype in config.datatypes:
            column_index.append(datatype.modality.path_column)
            column_index.append(datatype.modality.sharded_file_name_column)
            if datatype.modality.column is not None:
                column_index.append(datatype.modality.column)

        columns = [i for i in column_index if i in df.columns]
        orig_columns = [i for i in df.columns if i not in columns]
        columns.extend(list(orig_columns))
        return df[columns]

    @staticmethod
    def _merge_sharded_dataframes(paths_dataframes: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
        for path, df in paths_dataframes:
            df_name = get_path_filename(path)
            df.insert(loc=1, column='split_name', value=df_name)
        return pd.concat([d[1] for d in paths_dataframes], ignore_index=True)

    def _post_process_sharded_dataframes(
        self,
        split_suffix: str,
        config: Union[ShardedFilesDatasetConfig, ShardsDatasetConfig],
        paths_dataframes: list[tuple[str, pd.DataFrame]]
    ) -> pd.DataFrame:
        df = self._merge_sharded_dataframes(paths_dataframes)

        columns_to_rename = config.user_columns_to_rename
        if len(columns_to_rename) > 0:
            df.rename(columns=columns_to_rename, inplace=True)

        df = self._convert_sharded_columns_to_path_columns(split_suffix, config, df)
        df = self._rearrange_dataframe_columns(df, config)
        return df

    def read_shards(
        self,
        config: ShardsDatasetConfig,
        validate_columns: bool = True,
        workers: int = 1,
        progress_bar: bool = True,
    ) -> ShardsDatasetProcessor:
        """Creates ShardsDatasetProcessor dataset

        Parameters
        ----------
        config: ShardsDatasetConfig
            Config of ShardsDatasetConfig type
        validate_columns: bool = True
            Whether to check if columns in different csvs are matched
        workers: int = 1
            Number of parallel processes
        progress_bar: bool = True
            Whether to display the progress bar

        Returns
        -------
        ShardsDatasetProcessor
            Instance of ShardsDatasetProcessor dataset
        """
        dataset_path = config.path.rstrip("/")
        datafiles_ext_dot = '.' + config.datafiles_ext.lstrip(".")
        archive_ext_dot = '.' + config.archives_ext.lstrip(".")

        filepaths = self.filesystem.listdir(dataset_path)
        datafiles = [p for p in filepaths if p.endswith(datafiles_ext_dot)]
        archive_paths = [p for p in filepaths if p.endswith(archive_ext_dot)]
        if len(datafiles) == 0:
            raise ValueError("No datafiles in this path")

        # validate paths
        table_paths_set = set(datafiles)
        archive_paths_set = set(archive_paths)
        for filepath in table_paths_set:
            assert filepath.replace(datafiles_ext_dot, archive_ext_dot) in archive_paths_set, \
                f"File {filepath} has not associated archive"
        for filepath in archive_paths_set:
            assert filepath.replace(archive_ext_dot, datafiles_ext_dot) in table_paths_set, \
                f"Archive {filepath} has not associated data file"
        #

        paths_dataframes = self._read_and_validate_dataframes(
            datafiles, config, validate_columns, workers, progress_bar,
        )
        df = self._post_process_sharded_dataframes(archive_ext_dot, config, paths_dataframes)
        processor = ShardsDatasetProcessor(
            filesystem=self.filesystem,
            df=df,
            config=config
        )
        return processor

    def read_sharded_files(
        self,
        config: ShardedFilesDatasetConfig,
        validate_columns: bool = True,
        workers: int = 1,
        progress_bar: bool = True,
    ) -> ShardedFilesDatasetProcessor:
        """Creates ShardedFilesDatasetProcessor dataset

        Parameters
        ----------
        config: ShardedFilesDatasetConfig
            Config of ShardedFilesDatasetConfig type
        validate_columns: bool = True
            Whether to check if columns in different csvs are matched
        workers: int = 1
            Number of parallel processes
        progress_bar: bool = True
            Whether to display the progress bar

        Returns
        -------
        ShardedFilesDatasetProcessor
            Instance of ShardedFilesDatasetProcessor dataset
        """
        dataset_path = config.path.rstrip("/")
        datafiles_ext = config.datafiles_ext.lstrip(".")

        filepaths = self.filesystem.listdir(dataset_path)
        datafiles = [p for p in filepaths if p.endswith('.'+datafiles_ext)]
        if len(datafiles) == 0:
            raise ValueError("No datafiles in this path")

        # validate paths
        table_paths_set = set(datafiles)
        for filepath in table_paths_set:
            assert filepath.replace('.'+datafiles_ext, '') in filepaths, \
                f"File {filepath} has not associated folder"
        #

        paths_dataframes = self._read_and_validate_dataframes(
            datafiles, config, validate_columns, workers, progress_bar,
        )
        df = self._post_process_sharded_dataframes('', config, paths_dataframes)
        processor = ShardedFilesDatasetProcessor(
            filesystem=self.filesystem,
            df=df,
            config=config
        )
        return processor

    def read_files(
        self,
        config: FilesDatasetConfig,
    ) -> FilesDatasetProcessor:
        """Creates FilesDatasetProcessor dataset

        Parameters
        ----------
        config: FilesDatasetConfig
            Config of FilesDatasetConfig type

        Returns
        -------
        FilesDatasetProcessor
            Instance of FilesDatasetProcessor dataset
        """
        table_path = config.table_path.rstrip("/")
        df = self.filesystem.read_dataframe(table_path)

        required_columns = list(config.user_column2default_column.keys())
        column_set = set(df.columns.tolist())
        for col in required_columns:
            assert col in column_set, f'Expected table to have "{col}" column'

        # renaming columns
        columns_to_rename = config.user_columns_to_rename
        if len(columns_to_rename) > 0:
            df.rename(columns=columns_to_rename, inplace=True)

        # append paths to files
        for datatype in config.datatypes:
            if isinstance(datatype, FileDataType):
                path_col = datatype.modality.path_column
                df[path_col] = df[path_col].apply(lambda x: self.filesystem.join(config.base_path, x))

        return FilesDatasetProcessor(
            filesystem=self.filesystem,
            df=df,
            config=config
        )

    def read_from_config(  # type: ignore
        self,
        config: DatasetConfig,
        **kwargs
    ) -> DatasetProcessor:
        """Creates DatasetProcessor from config

        Parameters
        ----------
        config: DatasetConfig
            Config of DatasetConfig type
        **kwargs
            Parameters for read_shards, read_sharded_files, read_files methods

        Returns
        -------
        DatasetProcessor
            Instance of DatasetProcessor dataset
        """
        processor: DatasetProcessor
        if isinstance(config, ShardsDatasetConfig):
            processor = self.read_shards(config, **kwargs)
        elif isinstance(config, ShardedFilesDatasetConfig):
            processor = self.read_sharded_files(config, **kwargs)
        elif isinstance(config, FilesDatasetConfig):
            processor = self.read_files(config)
        else:
            raise ValueError(f"Unsupported config: {config}")
        return processor

    def from_df(self, config: DatasetConfig, df: pd.DataFrame) -> DatasetProcessor:
        """Creates DatasetProcessor from config and dataframe

        Parameters
        ----------
        config: DatasetConfig
            Config of DatasetConfig type
        df: pd.DataFrame
            Dataframe for DatasetProcessor.df

        Returns
        -------
        DatasetProcessor
            Instance of DatasetProcessor dataset
        """
        processor_class: type[DatasetProcessor]
        if isinstance(config, ShardsDatasetConfig):
            processor_class = ShardsDatasetProcessor
        elif isinstance(config, ShardedFilesDatasetConfig):
            processor_class = ShardedFilesDatasetProcessor
        elif isinstance(config, FilesDatasetConfig):
            processor_class = FilesDatasetProcessor
        else:
            raise ValueError(f"Unsupported config: {config}")

        return processor_class(
            filesystem=self.filesystem,
            config=config,
            df=df
        )
