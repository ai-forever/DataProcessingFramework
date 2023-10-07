from typing import Type

from DPF.modalities import MODALITIES
from DPF.datatypes import ShardedDataType, ColumnDataType
from DPF.configs import DatasetConfig, ShardsDatasetConfig, ShardedDatasetConfig, ShardedFilesDatasetConfig


class ShardedConfigFabric:

    @staticmethod
    def get_config_type(format_type: str) -> Type:
        if format_type == 'shards':
            return ShardsDatasetConfig
        elif format_type == 'sharded_files':
            return ShardedFilesDatasetConfig
        else:
            raise ValueError()

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