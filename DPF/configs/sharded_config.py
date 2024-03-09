from typing import Dict, List, Union

from DPF.datatypes import ColumnDataType, DataType, ShardedDataType

from ..modalities import ModalityName
from .dataset_config import DatasetConfig


class ShardedDatasetConfig(DatasetConfig):

    def __init__(
        self,
        path: str,
        datatypes: List[Union[ShardedDataType, ColumnDataType]],
        datafiles_ext: str = "csv",
    ):
        super().__init__(path)
        self._datatypes = datatypes
        self.datafiles_ext = datafiles_ext.lstrip('.')
        self._modality2datatype = {d.modality.name: d for d in datatypes}

        assert len({d.modality.name for d in datatypes}) == len(datatypes), \
            "More than one datatype with same modality is not supported"
        for data in self.datatypes:
            assert isinstance(data, (ColumnDataType, ShardedDataType))

    @property
    def datatypes(self) -> List[DataType]:
        return self._datatypes  # type: ignore

    @property
    def modality2datatype(self) -> Dict[ModalityName, DataType]:
        return self._modality2datatype  # type: ignore

    @property
    def user_column2default_column(self) -> Dict[str, str]:
        mapping = {}
        for data in self.datatypes:
            if isinstance(data, ColumnDataType):
                mapping[data.user_column_name] = data.column_name
            elif isinstance(data, ShardedDataType):
                mapping[data.user_basename_column_name] = data.modality.sharded_file_name_column
        return mapping

    def __repr__(self) -> str:
        s = "ShardedDatasetConfig(\n\t"
        s += f'path="{self.path}",\n\t'
        s += f'datafiles_ext="{self.datafiles_ext}",\n\t'
        s += 'datatypes=[\n\t\t'
        s += '\n\t\t'.join([str(i) for i in self.datatypes])
        s += '\n\t]'
        s += '\n)'
        return s
