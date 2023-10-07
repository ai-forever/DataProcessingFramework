from typing import Dict, Optional
import os
import pandas as pd
from tqdm import tqdm

from DPF.utils import get_file_extension
from DPF.utils.constants import ALLOWED_IMAGES_FORMATS
from DPF.processors.deprecated.images import ImagesProcessor
from .formatter import Formatter


class ImagesFormatter(Formatter):
    """
    Formatter for image datasets.
    Formatter is used to read and create a Processor class for a dataset.
    """

    def __init__(self, filesystem: str = "local", **filesystem_kwargs):
        super().__init__(filesystem, **filesystem_kwargs)

    def _postprocess_dataframe(self, df: pd.DataFrame):
        # TODO: do not create a copy of df, use inplace operations
        columns = ["image_name", "image_path", "data_format", "label"]
        columns = [i for i in columns if i in df.columns]
        return df[columns]

    def from_folder(
        self,
        directory: str,
        progress_bar: bool = True,
        allowed_image_formats: Optional[set] = None,
    ) -> ImagesProcessor:
        if allowed_image_formats is None:
            allowed_image_formats = ALLOWED_IMAGES_FORMATS
        allowed_image_formats = set(allowed_image_formats)

        directory = directory.rstrip("/")

        image_paths = []
        pbar = tqdm(disable=not progress_bar)
        for root, _, files in self.filesystem.walk(directory):
            for filename in files:
                pbar.update(1)
                file_ext = get_file_extension(filename)[1:]
                if file_ext in allowed_image_formats:
                    path = os.path.join(root, filename)
                    image_paths.append(path)

        return self.from_paths(
            image_paths=image_paths,
            check_image_extenstions=False,
            allowed_image_formats=allowed_image_formats,
        )

    def from_labeled_folder(
        self,
        directory: str,
        progress_bar: bool = True,
        allowed_image_formats: Optional[set] = None,
    ) -> ImagesProcessor:
        if allowed_image_formats is None:
            allowed_image_formats = ALLOWED_IMAGES_FORMATS
        allowed_image_formats = set(allowed_image_formats)

        directory = directory.rstrip("/")

        image_paths = []
        labels = []
        pbar = tqdm(disable=not progress_bar)
        for root, _, files in self.filesystem.walk(directory):
            for filename in files:
                pbar.update(1)
                file_ext = get_file_extension(filename)[1:]
                if file_ext in allowed_image_formats:
                    path = os.path.join(root, filename)
                    image_paths.append(path)
                    label = root.lstrip(directory + "/").split("/")[0]
                    labels.append(label)

        return self.from_paths(
            image_paths=image_paths,
            check_image_extenstions=False,
            additional_columns={"label": labels},
            allowed_image_formats=allowed_image_formats,
        )

    def from_paths(
        self,
        image_paths: list,
        check_image_extenstions: bool = True,
        additional_columns: Optional[Dict[str, list]] = None,
        allowed_image_formats: Optional[set] = None,
    ) -> ImagesProcessor:
        if allowed_image_formats is None:
            allowed_image_formats = ALLOWED_IMAGES_FORMATS
        allowed_image_formats = set(allowed_image_formats)

        image_paths_filtered = []
        image_names = []
        for path in image_paths:
            filename = os.path.basename(path)
            if check_image_extenstions:
                file_ext = get_file_extension(filename)[1:]
                if file_ext not in allowed_image_formats:
                    continue

            image_paths_filtered.append(path)
            image_names.append(filename)

        if additional_columns is None:
            additional_columns = {}

        df = pd.DataFrame(
            {
                "image_path": image_paths_filtered,
                "image_name": image_names,
                **additional_columns,
            }
        )
        df["data_format"] = "images_raw"
        df = self._postprocess_dataframe(df)

        return ImagesProcessor(filesystem=self.filesystem, df=df)
