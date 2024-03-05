from typing import Any, Dict, Union


# TODO(review) - логика работы непонятна совсем, для чего метод нужен, нужны пояснения + рефактор (выглядит как что-то ненужное)
# default identical preprocessing function for FilesDataset and ShardsDataset
def identical_preprocess_function(modality2data: Dict[str, Union[bytes, Any]], data: Dict[str, str]) -> Any:
    return modality2data, data


# identical collate function for pytorch dataloader
def identical_collate_fn(x: Any) -> Any:
    return x
