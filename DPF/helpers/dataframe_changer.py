from typing import List, Optional
import pandas as pd


class DataframeChanger:
    """
    DataframeChanger is used to change dataframes.
    """

    def __init__(self, filesystem, imagename_column, image_ext):
        self.filesystem = filesystem

        self.imagename_column = imagename_column
        self.image_ext = image_ext

    def _save_dataframe(self, df, path, **kwargs) -> Optional[str]:
        errname = None
        try:
            self.filesystem.save_dataframe(df, path, **kwargs)
        except Exception as err:
            errname = f"Error during saving file {path}: {err}"
        return errname

    def _rename_and_write_table(self, table_path, col2newcol) -> List[str]:
        df = self.filesystem.read_dataframe(table_path)
        df.rename(columns=col2newcol, inplace=True)

        errors = []
        errname = self._save_dataframe(df, table_path, index=False)
        if errname:
            errors.append(errname)

        return errors

    def _rename_and_write_table_mp(self, data):
        return self._rename_and_write_table(*data)

    def _delete_and_write_table(self, table_path, columns_to_delete) -> List[str]:
        df = self.filesystem.read_dataframe(table_path)
        df.drop(columns=columns_to_delete, inplace=True)

        errors = []
        errname = self._save_dataframe(df, table_path, index=False)
        if errname:
            errors.append(errname)

        return errors

    def _delete_and_write_table_mp(self, data):
        return self._delete_and_write_table(*data)

    def _merge_and_write_table(
        self, table_path, df_to_add, overwrite_columns=True
    ) -> List[str]:
        if self.image_ext:
            image_ext = self.image_ext.lstrip(".")
            df_to_add["image_name"] = df_to_add["image_name"].str.slice(
                0, -len(image_ext) - 1
            )
        df_to_add.rename(columns={"image_name": self.imagename_column}, inplace=True)

        df = self.filesystem.read_dataframe(table_path)
        columns = [i for i in df.columns if i != self.imagename_column]
        columns_to_add = [i for i in df_to_add.columns if i != self.imagename_column]
        columns_intersection = set(columns).intersection(set(columns_to_add))
        if overwrite_columns:
            df.drop(columns=list(columns_intersection), inplace=True)
        else:
            df_to_add.drop(columns=list(columns_intersection), inplace=True)

        errors = []
        if df_to_add.shape[1] > 1:
            image_names_orig = set(df[self.imagename_column])
            shape_orig = len(df)
            df = pd.merge(df, df_to_add, on=self.imagename_column)

            errname = None
            if len(df) != shape_orig:
                errname = (
                    f"Shape of dataframe {table_path} changed after merging. "
                    "Skipping this dataframe. Check for errors"
                )
                print("[WARNING]", errname)
            elif set(df[self.imagename_column]) != image_names_orig:
                errname = (
                    f"Image names from dataframe {table_path} changed after merging. "
                    "Skipping this dataframe. Check for errors"
                )
                print("[WARNING]", errname)
            else:
                errname = self._save_dataframe(df, table_path, index=False)
            if errname:
                errors.append(errname)

        return errors

    def _merge_and_write_table_mp(self, data):
        return self._merge_and_write_table(*data)
