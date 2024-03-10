import os
from typing import Optional, Union

from DPF.datatypes import ColumnDataType, DataType, FileDataType
from DPF.modalities import MODALITIES, ModalityName

from .dataset_config import DatasetConfig


class FilesDatasetConfig(DatasetConfig):

    def __init__(
        self,
        path: str,
        datatypes: list[Union[FileDataType, ColumnDataType]],
    ):
        super().__init__(path)
        self.table_path = path
        self.base_path = os.path.dirname(self.table_path)
        self._datatypes = datatypes
        self._modality2datatype = {d.modality.name: d for d in datatypes}

        assert len({d.modality.name for d in datatypes}) == len(datatypes), \
            "More than one datatype with same modality is not supported"
        for data in self.datatypes:
            assert isinstance(data, (ColumnDataType, FileDataType))

    @property
    def datatypes(self) -> list[DataType]:
        return self._datatypes  # type: ignore

    @property
    def modality2datatype(self) -> dict[ModalityName, DataType]:
        return self._modality2datatype  # type: ignore

    @property
    def user_column2default_column(self) -> dict[str, str]:
        mapping = {}
        for data in self.datatypes:
            if isinstance(data, ColumnDataType):
                mapping[data.user_column_name] = data.column_name
            elif isinstance(data, FileDataType):
                mapping[data.user_path_column_name] = data.modality.path_column
        return mapping

    @classmethod
    def from_path_and_columns(
        cls,
        path: str,
        image_path_col: Optional[str] = None,
        video_path_col: Optional[str] = None,
        caption_col: Optional[str] = None,
    ) -> "FilesDatasetConfig":
        datatypes: list[Union[FileDataType, ColumnDataType]] = []
        if image_path_col:
            datatypes.append(FileDataType(MODALITIES['image'], image_path_col))
        if video_path_col:
            datatypes.append(FileDataType(MODALITIES['video'], video_path_col))
        if caption_col:
            datatypes.append(ColumnDataType(MODALITIES['text'], caption_col))
        assert len(datatypes) > 0, "At least one modality should be provided"
        return cls(path, datatypes)

    def __repr__(self) -> str:
        s = "FilesDatasetConfig(\n\t"
        s += f'table_path="{self.table_path}",\n\t'
        s += 'datatypes=[\n\t\t'
        s += '\n\t\t'.join([str(i) for i in self.datatypes])
        s += '\n\t]'
        s += '\n)'
        return s
