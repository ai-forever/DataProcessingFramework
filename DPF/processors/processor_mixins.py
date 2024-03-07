from typing import Union

import pandas as pd

from DPF.configs import DatasetConfig
from DPF.modalities import MODALITIES
from DPF.transforms import BaseFilesTransforms


class ApplyTransformProcessorMixin:
    config: DatasetConfig
    _df: pd.DataFrame

    def apply_transform(self, transforms: Union[BaseFilesTransforms]) -> None:
        assert transforms.modality in self.config.modality2datatype

        filepath_column = MODALITIES[transforms.modality].path_column
        filepaths = self._df[filepath_column].tolist()

        metadata_lists = None
        if len(transforms.required_metadata) > 0:
            metadata_lists = {
                col: self._df[col].tolist()
                for col in transforms.required_metadata
            }

        transformed_metadata = transforms.run(filepaths, metadata_lists=metadata_lists)
        for data in transformed_metadata:
            data.metadata[filepath_column] = data.filepath
        df_to_merge = pd.DataFrame([data.metadata for data in transformed_metadata])

        # drop metadata columns from original df to replace them
        self._df.drop(columns=transforms.metadata_to_change, errors='ignore', inplace=True)

        self._df = pd.merge(self._df, df_to_merge, on=filepath_column, how='left')
