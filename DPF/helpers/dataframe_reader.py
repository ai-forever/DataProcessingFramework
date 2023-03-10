from typing import Set
import pandas as pd

from DPF.filesystems import FileSystem


class DataframeReader:
    """
    DataframeReader is used to read and preprocess dataframes
    """

    def __init__(
            self,
            filesystem: FileSystem,
            read_params: dict,
            df_needed_columns: Set[str],
            check_same_columns: bool = True,
        ):
        self.filesystem = filesystem
        self.read_params = read_params

        self.check_same_columns = check_same_columns
        self.df_needed_columns = df_needed_columns

    def _add_base_columns(
            self,
            df: pd.DataFrame,
            filepath: str,
            caption_column: str,
            imagename_column: str,
            image_ext: str
        ):
        if self.check_same_columns:
            assert set(df.columns) == self.df_needed_columns, \
                f'Dataframe {filepath} have different columns. ' \
                f'Expected {self.df_needed_columns}, got {set(df.columns)}'
        assert imagename_column in df.columns, \
            f'Dataframe {filepath} does not have "{imagename_column}" column'
        assert caption_column in df.columns, \
            f'Dataframe {filepath} does not have "{caption_column}" column'

        df['table_path'] = filepath
        df['caption'] = df[caption_column]
        df['image_name'] = df[imagename_column]
        if image_ext:
            image_ext = image_ext.lstrip('.')
            df['image_name'] += '.'+image_ext

    def load_shards_df(self, filepath: str) -> pd.DataFrame:
        datafiles_ext = self.read_params["datafiles_ext"]
        archive_ext = self.read_params["archive_ext"]
        image_ext = self.read_params["image_ext"]
        caption_column = self.read_params["caption_column"]
        imagename_column = self.read_params["imagename_column"]

        df = self.filesystem.read_dataframe(filepath)

        self._add_base_columns(df, filepath, caption_column, imagename_column, image_ext)

        df['archive_path'] = df['table_path'].str.rstrip(datafiles_ext)+archive_ext
        df['image_path'] = df['archive_path']+'/'+df['image_name']
        return df

    def load_raw_df(self, filepath: str) -> pd.DataFrame:
        datafiles_ext = self.read_params["datafiles_ext"]
        image_ext = self.read_params["image_ext"]
        caption_column = self.read_params["caption_column"]
        imagename_column = self.read_params["imagename_column"]

        df = self.filesystem.read_dataframe(filepath)

        self._add_base_columns(df, filepath, caption_column, imagename_column, image_ext)

        df['image_path'] = df['table_path'].str.slice(0,-(len(datafiles_ext)+1)) \
                           + '/' + df['image_name']
        return df
