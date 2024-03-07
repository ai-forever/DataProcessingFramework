from typing import Any, Dict, List, Union

from DPF.datatypes import ColumnDataType, FileDataType, ShardedDataType
from DPF.modalities import ModalityName
from DPF.types import ModalityToDataMapping


# TODO(review) - логика работы непонятна совсем, для чего метод нужен, нужны пояснения + рефактор (выглядит как что-то ненужное)
# default identical preprocessing function for FilesDataset and ShardsDataset
def identical_preprocess_function(modality2data: ModalityToDataMapping, metadata: Dict[str, str]) -> Any:
    return modality2data, metadata


# identical collate function for pytorch dataloader
def identical_collate_fn(x: Any) -> Any:
    return x


def get_paths_columns_to_modality_mapping(
    datatypes: List[Union[ShardedDataType, FileDataType]]
) -> Dict[str, ModalityName]:
    return {
        datatype.modality.path_column: datatype.modality.name
        for datatype in datatypes
    }


def get_columns_to_modality_mapping(
    datatypes: List[ColumnDataType]
) -> Dict[str, ModalityName]:
    return {
        datatype.column_name: datatype.modality.name
        for datatype in datatypes
    }
