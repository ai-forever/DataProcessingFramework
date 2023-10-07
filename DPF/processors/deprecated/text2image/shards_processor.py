from typing import Optional
import random
import tarfile
import io
import pandas as pd
from PIL import Image

from DPF.filesystems import FileSystem
from DPF.processors.deprecated.text2image.t2i_processor import T2IProcessor


class ShardsProcessor(T2IProcessor):
    """
    Class that describes all interactions with text2image dataset
    in shards format (archives with images and dataframes).
    It is recommended to use T2IFormatter to create ShardsProcessor
    instead of directly initialiasing a ShardsProcessor class.
    """

    def __init__(
        self,
        filesystem: FileSystem,
        df: pd.DataFrame,
        dataset_path: str,
        archive_ext: str,
        datafiles_ext: str,
        imagename_column: str,
        caption_column: str,
        image_ext: str,
    ):
        super().__init__(
            filesystem,
            df,
            dataset_path,
            datafiles_ext,
            imagename_column,
            caption_column,
            image_ext,
        )

        self.archive_ext = archive_ext.lstrip(".")

    def get_random_samples(
        self, df: Optional[pd.DataFrame] = None, n: int = 1, from_tars: int = 1
    ) -> list:
        """
        Get N random samples from dataset

        Parameters
        ----------
        df: pd.DataFrame | None
            DataFrame to sample from. If none, processor.df is used
        n: int = 1
            Number of samples to return
        from_tars: int = 1
            Number of archives to sample from

        Returns
        -------
        list
            List of tuples with PIL images and dataframe data
        """

        if df is None:
            df = self.df

        archives = random.sample(df["archive_path"].unique().tolist(), from_tars)
        df_samples = df[df["archive_path"].isin(archives)].sample(n)

        archive_to_samples = df_samples.groupby("archive_path").apply(
            lambda x: x.to_dict("records")
        )

        samples = []
        for archive_path, data in archive_to_samples.to_dict().items():
            tar_bytes = self.filesystem.read_file(archive_path, binary=True)
            with tarfile.open("r", fileobj=tar_bytes) as tar:
                for item in data:
                    filename = item[self.imagename_column]
                    img = Image.open(io.BytesIO(tar.extractfile(filename).read()))
                    samples.append((img, item))
        return samples
