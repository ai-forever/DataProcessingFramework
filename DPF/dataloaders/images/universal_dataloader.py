from torch.utils.data import DataLoader

from DPF.filesystems.filesystem import FileSystem
from .utils import default_preprocess
from .raw_dataset import RawDataset
from .shards_dataset import ShardsDataset


FORMAT_TO_DATASET = {
    "images_raw": RawDataset,
    "raw": RawDataset,
    "shards": ShardsDataset,
}


class UniversalT2IDataloader:
    """
    Universal text2image dataset class
    """

    def __init__(
        self,
        filesystem: FileSystem,
        df,
        cols_to_return=None,
        preprocess_f=default_preprocess,
        return_none_on_error: bool = False,
        **dataloader_kwargs,
    ):
        if cols_to_return is None:
            cols_to_return = []
        self.filesystem = filesystem
        self.df = df
        self.df_formats = df["data_format"].unique().tolist()
        assert all(
            f in FORMAT_TO_DATASET for f in self.df_formats
        ), "Unknown data format in dataloader"
        self.cols_to_return = cols_to_return
        self.preprocess_f = preprocess_f
        self.return_none_on_error = return_none_on_error
        self.dataloader_kwargs = dataloader_kwargs
        self.len = None

    def test(self):
        for data_format in self.df_formats:
            dataset_class = FORMAT_TO_DATASET[data_format]
            dataset = dataset_class(
                self.filesystem,
                self.df[self.df["data_format"] == data_format],
                self.cols_to_return,
                self.preprocess_f,
                self.return_none_on_error
            )
            print(f'"{data_format}" dataset created')
            dataloader = DataLoader(dataset, **self.dataloader_kwargs)
            print(f'"{data_format}" dataloader created')
            for _ in dataloader:
                break
            print(f'"{data_format}" iteration tested. Format "{data_format}" is ok!')

    def __len__(self):
        if "batch_size" in self.dataloader_kwargs:
            bs = self.dataloader_kwargs["batch_size"]
        else:
            bs = 1

        format_counts = self.df["data_format"].value_counts().to_dict()
        batched_len = sum(
            count // bs if count % bs == 0 else (count // bs) + 1
            for count in format_counts.values()
        )
        return batched_len

    def __iter__(self):
        for data_format in self.df_formats:
            dataset_class = FORMAT_TO_DATASET[data_format]
            dataset = dataset_class(
                self.filesystem,
                self.df[self.df["data_format"] == data_format],
                self.cols_to_return,
                self.preprocess_f,
            )
            dataloader = DataLoader(dataset, **self.dataloader_kwargs)
            for item in dataloader:
                yield item
