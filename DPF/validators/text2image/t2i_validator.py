from typing import Dict, List
import pandas as pd
import numpy as np

from DPF.filesystems.filesystem import FileSystem
from DPF.validators.utils import get_duplicated_elements
from DPF.validators import Validator


class T2IValidator(Validator):
    """
    Validator is used to check the text2image dataset for compliance with a specific storage format
    """

    def __init__(
        self,
        filesystem: FileSystem,
        csv_columns: List[str],
        image_name_col: str = "image_name",
        caption_column: str = "caption",
        validate_captions: bool = True,
    ):
        self.filesystem = filesystem
        self.caption_column = caption_column

        self.csv_columns = csv_columns
        self.csv_columns_set = set(self.csv_columns)
        self.image_name_col = image_name_col

        self.validate_captions = validate_captions

    def validate_caption_df(self, df: pd.DataFrame) -> (dict, Dict[str, int]):
        errors = {}
        error2count = {}

        df_caption_isna = df[self.caption_column].isna()
        df_caption_small = df[self.caption_column].str.strip().str.len() <= 2
        if df_caption_isna.any():
            errors["ok"] = False
            errname = "provided caption column has None values"
            errors[errname] = sum(df_caption_isna)
            error2count[errname] = sum(df_caption_isna)
        elif df_caption_small.any():
            errors["ok"] = False
            errname = "provided caption column has values with length less than 2"
            errors[errname] = sum(df_caption_small)
            error2count[errname] = sum(df_caption_small)

        return errors, error2count

    def validate_df(self, df: pd.DataFrame) -> (dict, Dict[str, int]):
        """
        Validates a dataframe. Checks required columns, duplicates and (optionally) captions.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe to validate

        Returns
        -------
        (dict, Dict[str, int])
            Dictionary with errors description and dictionary with number of occured errors
        """

        errors = {"ok": True}
        error2count = {}

        missed_columns = self.csv_columns_set.difference(set(df.columns))
        if len(missed_columns) > 0:
            errname = "missed columns"
            errors[errname] = list(missed_columns)
            error2count[errname] = 1
            errors["ok"] = False

        image_names_in_csv = df[self.image_name_col]
        duplicated_images_in_csv = np.unique(
            get_duplicated_elements(image_names_in_csv)
        )
        if len(duplicated_images_in_csv) > 0:
            errname = "duplicated images in csv"
            errors[errname] = list(duplicated_images_in_csv)
            error2count[errname] = len(errors[errname])
            errors["ok"] = False

        if self.validate_captions:
            errors_caption, error2count_caption = self.validate_caption_df(df)
            errors.update(errors_caption)
            error2count.update(error2count_caption)

        return errors, error2count
