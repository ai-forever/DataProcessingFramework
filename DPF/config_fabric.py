from typing import Type, Optional

from DPF.modalities import MODALITIES
from DPF.datatypes import ShardedDataType, ColumnDataType, FileDataType
from DPF.configs import (
    DatasetConfig, ShardsDatasetConfig, ShardedDatasetConfig, ShardedFilesDatasetConfig,
    FilesDatasetConfig
)


class ShardedConfigFabric:

    @staticmethod
    def get_config_type(format_type: str) -> Type:
        if format_type == 'shards':
            return ShardsDatasetConfig
        elif format_type == 'sharded_files':
            return ShardedFilesDatasetConfig
        else:
            raise ValueError()

    def create_files_config(
        self,
        data_path: str,
        image_path_col: Optional[str] = None,
        caption_col: Optional[str] = None,
    ) -> FilesDatasetConfig:
        datatypes = []
        if image_path_col:
            datatypes.append(FileDataType(MODALITIES['image'], image_path_col))
        if caption_col:
            datatypes.append(ColumnDataType(MODALITIES['text'], caption_col))
        assert len(datatypes) > 0
        return FilesDatasetConfig(data_path, datatypes)

    def create_t2i_config(
        self,
        path: str,
        format_type: str,
        image_name_col: str = 'image_name',
        caption_col: str = 'caption',
        **kwargs
    ) -> ShardedDatasetConfig:
        image_datatype = ShardedDataType(MODALITIES['image'], image_name_col)
        text_datatype = ColumnDataType(MODALITIES['text'], caption_col)

        config_type = self.get_config_type(format_type)
        return config_type(path, [image_datatype, text_datatype], **kwargs)

    def create_t2v_config(
        self,
        path: str,
        format_type: str,
        video_name_col: str = 'video_name',
        caption_col: str = 'caption',
        **kwargs
    ) -> ShardedDatasetConfig:
        video_datatype = ShardedDataType(MODALITIES['video'], video_name_col)
        text_datatype = ColumnDataType(MODALITIES['text'], caption_col)

        config_type = self.get_config_type(format_type)
        return config_type(path, [video_datatype, text_datatype], **kwargs)