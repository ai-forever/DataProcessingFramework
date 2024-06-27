from typing import Optional, Union

from DPF.datatypes import ColumnDataType, ShardedDataType
from DPF.modalities import MODALITIES

from .sharded_config import ShardedDatasetConfig


class ShardedFilesDatasetConfig(ShardedDatasetConfig):
    """Config for ShardedFiles dataset type"""

    def __init__(
        self,
        path: str,
        datatypes: list[Union[ShardedDataType, ColumnDataType]],
        datafiles_ext: str = "csv",
    ):
        """
        Parameters
        ----------
        path: str
            Path to directory with shards
        datatypes: list[Union[ShardedDataType, ColumnDataType]]
            List of datatypes in dataset
        datafiles_ext: str = "csv"
            Extension of files with metadata in shards
        """
        super().__init__(path, datatypes, datafiles_ext)

    @classmethod
    def from_path_and_columns(
        cls,
        path: str,
        image_name_col: Optional[str] = None,
        video_name_col: Optional[str] = None,
        audio_name_col: Optional[str] = None,
        text_col: Optional[str] = None,
        datafiles_ext: str = "csv",
    ) -> "ShardedFilesDatasetConfig":
        """
        Parameters
        ----------
        path: str
            Path to directory with shards
        image_name_col: Optional[str] = None
            Name of column with image filenames in shard
        video_name_col: Optional[str] = None
            Name of column with video filenames in shard
        audio_name_col: Optional[str] = None
            Name of column with audio filenames in shard
        text_col: Optional[str] = None
            Name of column with text
        datafiles_ext: str = "csv"
            Extension of files with metadata in shards

        Returns
        -------
        ShardedFilesDatasetConfig
            Instance of itself
        """
        datatypes: list[Union[ShardedDataType, ColumnDataType]] = []
        if image_name_col:
            datatypes.append(ShardedDataType(MODALITIES['image'], image_name_col))
        if video_name_col:
            datatypes.append(ShardedDataType(MODALITIES['video'], video_name_col))
        if audio_name_col:
            datatypes.append(ShardedDataType(MODALITIES['audio'], audio_name_col))
        if text_col:
            datatypes.append(ColumnDataType(MODALITIES['text'], text_col))
        assert len(datatypes) > 0, "At least one modality should be provided"
        return cls(path, datatypes, datafiles_ext=datafiles_ext)
