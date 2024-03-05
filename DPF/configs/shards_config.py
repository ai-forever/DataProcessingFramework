from typing import List, Optional, Union

from DPF.datatypes import ColumnDataType, ShardedDataType
from DPF.modalities import MODALITIES

from .sharded_config import ShardedDatasetConfig


class ShardsDatasetConfig(ShardedDatasetConfig):

    def __init__(
        self,
        path: str,
        datatypes: List[Union[ShardedDataType, ColumnDataType]],
        archives_ext: str = "tar",
        datafiles_ext: str = "csv",
    ):
        super().__init__(path, datatypes, datafiles_ext)
        self.archives_ext = archives_ext.lstrip('.')

    @classmethod
    def from_modalities(
        cls,
        path: str,
        image_name_col: Optional[str] = None,
        video_name_col: Optional[str] = None,
        caption_col: Optional[str] = None,
        archives_ext: str = "tar",
        datafiles_ext: str = "csv",
    ):
        datatypes = []
        if image_name_col:
            datatypes.append(ShardedDataType(MODALITIES['image'], image_name_col))
        if video_name_col:
            datatypes.append(ShardedDataType(MODALITIES['video'], video_name_col))
        if caption_col:
            datatypes.append(ColumnDataType(MODALITIES['text'], caption_col))
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
