from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod

from DPF.modalities import MODALITIES
from DPF.datatypes import DataType, ShardedDataType, ColumnDataType


class DatasetConfig:

    def __init__(
        self,
        datatypes: List[DataType],
    ):
        assert len(set([d.modality.key for d in datatypes])) == len(datatypes)
        self.datatypes = datatypes

    @property
    @abstractmethod
    def modality2datatype(self) -> Dict[str, DataType]:
        pass

    @property
    @abstractmethod
    def columns_mapping(self) -> Dict[str, str]:
        pass

    def __repr__(self) -> str:
        s = "DatasetConfig(\n\t"
        s += 'datatypes=[\n\t\t'
        s += '\n\t\t'.join([str(i) for i in self.datatypes])
        s += '\n\t]'
        s += '\n)'
        return s


class ShardedDatasetConfig(DatasetConfig):

    def __init__(
        self,
        path: str,
        datatypes: List[Union[ShardedDataType, ColumnDataType]],
        datafiles_ext: str = "csv",
    ):
        super().__init__(datatypes)
        self.path = path.rstrip('/')
        self.datatypes = datatypes
        self.datafiles_ext = datafiles_ext.lstrip('.')
        self._modality2datatype = {d.modality.key: d for d in datatypes}
        self.validate_datatypes()

    def validate_datatypes(self):
        for data in self.datatypes:
            assert isinstance(data, (ColumnDataType, ShardedDataType))

    @property
    def modality2datatype(self) -> Dict[str, DataType]:
        return self._modality2datatype

    @property
    def columns_mapping(self) -> Dict[str, str]:
        mapping = {}
        for data in self.datatypes:
            if isinstance(data, ColumnDataType):
                mapping[data.user_column_name] = data.modality.column
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

    def __repr__(self) -> str:
        s = "ShardedDatasetConfig(\n\t"
        s += f'path="{self.path}",\n\t'
        s += f'archives_ext="{self.archives_ext}",\n\t'
        s += f'datafiles_ext="{self.datafiles_ext}",\n\t'
        s += 'datatypes=[\n\t\t'
        s += '\n\t\t'.join([str(i) for i in self.datatypes])
        s += '\n\t]'
        s += '\n)'
        return s


class ShardedFilesDatasetConfig(ShardedDatasetConfig):

    def __init__(
        self,
        path: str,
        datatypes: List[Union[ShardedDataType, ColumnDataType]],
        datafiles_ext: str = "csv",
    ):
        super().__init__(path, datatypes, datafiles_ext)
