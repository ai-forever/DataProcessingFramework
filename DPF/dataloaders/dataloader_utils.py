from typing import Dict, Any


# TODO(review) - логика работы непонятна совсем, для чего метод нужен, нужны пояснения + рефактор (выглядит как что-то ненужное)
# default identical preprocessing function for FilesDataset and ShardsDataset
def identical_preprocess_function(column2bytes: Dict[str, bytes], data: Dict[str, str]) -> Any:
    return column2bytes, data


# identical collate function for pytorch dataloader
def identical_collate_fn(x):
    return x
