from typing import List, Optional, Dict
import pandas as pd
from DPF.filesystems import FileSystem
from DPF.configs import DatasetConfig
from tqdm.contrib.concurrent import thread_map


class DataFramesChanger:

    def __init__(
        self,
        datafile_paths: List[str],
        filesystem: FileSystem,
        config: DatasetConfig
    ):
        self.datafile_paths = datafile_paths
        self.filesystem = filesystem
        self.config = config

    def _save_dataframe(self, df, path, **kwargs) -> Optional[str]:
        errname = None
        try:
            self.filesystem.save_dataframe(df, path, **kwargs)
        except Exception as err:
            errname = f"Error during saving file {path}: {err}"
        return errname

    def validate_path_for_delete(self, columns_to_delete: List[str], path: str):
        df = self.filesystem.read_dataframe(path)
        for col in columns_to_delete:
            assert col in df.columns, f'Dataframe {path} dont have "{col}" column'

    def delete_columns_for_path(self, columns_to_delete: List[str], path: str) -> Optional[str]:
        df = self.filesystem.read_dataframe(path)
        df.drop(columns=columns_to_delete, inplace=True)
        return self._save_dataframe(df, path, index=False)

    def delete_columns(
        self,
        columns_to_delete: List[str],
        max_threads: int = 16,
        pbar: bool = True
    ) -> List[str]:
        # validate all files before modifying them
        thread_map(
            lambda p: self.validate_path_for_delete(columns_to_delete, p),
            self.datafile_paths,
            max_workers=max_threads,
            disable=not pbar
        )
        # modify datafiles
        errors = thread_map(
            lambda p: self.delete_columns_for_path(columns_to_delete, p),
            self.datafile_paths,
            max_workers=max_threads,
            disable=not pbar
        )
        return [err for err in errors if err is not None]

    def validate_path_for_rename(self, column_map: Dict[str, str], path: str):
        df = self.filesystem.read_dataframe(path)
        for col_old, col_new in column_map.items():
            assert col_old in df.columns, f'Dataframe {path} dont have "{col_old}" column'
            assert col_new not in df.columns, f'Dataframe {path} already have "{col_new}" column'

    def rename_columns_for_path(self, column_map: Dict[str, str], path: str) -> Optional[str]:
        df = self.filesystem.read_dataframe(path)
        df.rename(columns=column_map, inplace=True)
        return self._save_dataframe(df, path, index=False)

    def rename_columns(
        self,
        column_map: Dict[str, str],
        max_threads: int = 16,
        pbar: bool = True
    ) -> List[str]:
        # validate all files before modifying them
        thread_map(
            lambda p: self.validate_path_for_rename(column_map, p),
            self.datafile_paths,
            max_workers=max_threads,
            disable=not pbar
        )
        # modify datafiles
        errors = thread_map(
            lambda p: self.rename_columns_for_path(column_map, p),
            self.datafile_paths,
            max_workers=max_threads,
            disable=not pbar
        )
        return [err for err in errors if err is not None]

    def validate_path_for_update(
        self,
        key_column: str,
        df_new: List[dict],
        path: str
    ):
        df_new = pd.DataFrame(df_new)
        df_old = self.filesystem.read_dataframe(path)
        assert key_column in df_old.columns, f'Dataframe {path} dont have "{key_column}" column'
        assert set(df_old[key_column]) == set(df_new[key_column]), \
            f'Dataframe {path} has different values in "{key_column}"'

        duplicates = df_old[df_old[key_column].duplicated()][key_column].tolist()
        assert len(duplicates) == 0, f'Dataframe {path} has duplicates in "{key_column}" column: {duplicates}'

        duplicates = df_new[df_new[key_column].duplicated()][key_column].tolist()
        assert len(duplicates) == 0, f'New dataframe for {path} has duplicates in "{key_column}" column: {duplicates}'

        assert len(df_old) == len(df_new), f'Length of {path} dataframe is changed'

    def update_columns_for_path(
        self,
        key_column: str,
        df_new: List[dict],
        path: str
    ) -> Optional[str]:
        df_new = pd.DataFrame(df_new)
        df_old = self.filesystem.read_dataframe(path)

        columns_to_add = [i for i in df_new.columns if i != key_column]
        columns_intersection = set(df_old.columns).intersection(set(columns_to_add))

        if len(columns_intersection) > 0:
            df_old.drop(columns=list(columns_intersection), inplace=True)

        df = pd.merge(df_old, df_new, on=key_column)
        return self._save_dataframe(df, path, index=False)

    def update_columns(
        self,
        key_column: str,
        path2df: Dict[str, List[dict]],
        max_threads: int = 16,
        pbar: bool = True
    ) -> List[str]:
        # validate all files before modifying them
        thread_map(
            lambda p: self.validate_path_for_update(key_column, path2df[p], p),
            list(path2df.keys()),
            max_workers=max_threads,
            disable=not pbar
        )
        # modify datafiles
        errors = thread_map(
            lambda p: self.update_columns_for_path(key_column, path2df[p], p),
            list(path2df.keys()),
            max_workers=max_threads,
            disable=not pbar
        )
        return [err for err in errors if err is not None]
