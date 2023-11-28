from typing import List, Dict, Optional, Union

from DPF.datatypes import DataType, ColumnDataType, FileDataType
from .dataset_config import DatasetConfig
from ..modalities import MODALITIES


class FilesDatasetConfig(DatasetConfig):

    def __init__(
        self,
        table_path: str,
        datatypes: List[Union[FileDataType, ColumnDataType]],
    ):
        super().__init__(datatypes)
        self.table_path = table_path.rstrip('/')
        self.datatypes = datatypes
        self._modality2datatype = {d.modality.key: d for d in datatypes}
        self.validate_datatypes()

    def validate_datatypes(self):
        for data in self.datatypes:
            assert isinstance(data, (ColumnDataType, FileDataType))

    @property
    def modality2datatype(self) -> Dict[str, DataType]:
        return self._modality2datatype

    @property
    def columns_mapping(self) -> Dict[str, str]:
        mapping = {}
        for data in self.datatypes:
            if isinstance(data, ColumnDataType):
                mapping[data.user_column_name] = data.modality.column
            elif isinstance(data, FileDataType):
                mapping[data.user_path_column_name] = data.modality.path_column
        return mapping

    @classmethod
    def from_modalities(
        cls,
        path: str,
        image_path_col: Optional[str] = None,
        video_path_col: Optional[str] = None,
        caption_col: Optional[str] = None,
        datafiles_ext: str = "csv",
    ):
        datatypes = []
        if image_path_col:
            datatypes.append(FileDataType(MODALITIES['image'], image_path_col))
        if video_path_col:
            datatypes.append(FileDataType(MODALITIES['video'], video_path_col))
        if caption_col:
            datatypes.append(ColumnDataType(MODALITIES['text'], caption_col))
        assert len(datatypes) > 0, "At least one modality should be provided"
        return cls(path, datatypes, datafiles_ext=datafiles_ext)

    def __repr__(self) -> str:
        s = "FilesDatasetConfig(\n\t"
        s += f'table_path="{self.table_path}",\n\t'
        s += 'datatypes=[\n\t\t'
        s += '\n\t\t'.join([str(i) for i in self.datatypes])
        s += '\n\t]'
        s += '\n)'
        return s