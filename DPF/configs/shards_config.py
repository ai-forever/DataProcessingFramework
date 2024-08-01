from typing import Optional, Union

from DPF.datatypes import ColumnDataType, ShardedDataType
from DPF.modalities import MODALITIES

from .sharded_config import ShardedDatasetConfig


class ShardsDatasetConfig(ShardedDatasetConfig):
    """Config for Shards dataset type"""

    def __init__(
        self,
        path: str,
        datatypes: list[Union[ShardedDataType, ColumnDataType]],
        archives_ext: str = "tar",
        datafiles_ext: str = "csv",
    ):
        """
        Parameters
        ----------
        path: str
            Path to directory with shards
        datatypes: list[Union[ShardedDataType, ColumnDataType]]
            List of datatypes in dataset
        archives_ext: str = "tar"
            Extension of archives in shards
        datafiles_ext: str = "csv"
            Extension of files with metadata in shards
        """
        super().__init__(path, datatypes, datafiles_ext)
        self.archives_ext = archives_ext.lstrip('.')

    @classmethod
    def from_path_and_columns(
        cls,
        path: str,
        image_name_col: Optional[str] = None,
        video_name_col: Optional[str] = None,
        audio_name_col: Optional[str] = None,
        text_col: Optional[str] = None,
        archives_ext: str = "tar",
        datafiles_ext: str = "csv",
    ) -> "ShardsDatasetConfig":
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
        archives_ext: str = "tar"
            Extension of archives in shards
        datafiles_ext: str = "csv"
            Extension of files with metadata in shards

        Returns
        -------
        ShardsDatasetConfig
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
        return cls(path, datatypes, archives_ext=archives_ext, datafiles_ext=datafiles_ext)

    def __repr__(self) -> str:
        s = "ShardsDatasetConfig(\n\t"
        s += f'path="{self.path}",\n\t'
        s += f'archives_ext="{self.archives_ext}",\n\t'
        s += f'datafiles_ext="{self.datafiles_ext}",\n\t'
        s += 'datatypes=[\n\t\t'
        s += '\n\t\t'.join([str(i) for i in self.datatypes])
        s += '\n\t]'
        s += '\n)'
        return s
