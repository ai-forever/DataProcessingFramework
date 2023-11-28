from typing import List, Dict, Optional, Union

from DPF.modalities import MODALITIES
from DPF.datatypes import ShardedDataType, ColumnDataType
from .sharded_config import ShardedDatasetConfig


class ShardedFilesDatasetConfig(ShardedDatasetConfig):

    def __init__(
        self,
        path: str,
        datatypes: List[Union[ShardedDataType, ColumnDataType]],
        datafiles_ext: str = "csv",
    ):
        super().__init__(path, datatypes, datafiles_ext)

    @classmethod
    def from_modalities(
        cls,
        path: str,
        image_name_col: Optional[str] = None,
        video_name_col: Optional[str] = None,
        caption_col: Optional[str] = None,
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
        return cls(path, datatypes, datafiles_ext=datafiles_ext)