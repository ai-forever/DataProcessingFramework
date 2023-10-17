from typing import List, Tuple, Union
import pandas as pd
from tqdm.contrib.concurrent import thread_map

from DPF.filesystems import FileSystem, LocalFileSystem, S3FileSystem
from DPF.datatypes import ShardedDataType, ColumnDataType
from DPF.configs import DatasetConfig, ShardedDatasetConfig, ShardsDatasetConfig, ShardedFilesDatasetConfig
from DPF.processors import (
    DatasetProcessor, ShardedDatasetProcessor, ShardsDatasetProcessor, ShardedFilesDatasetProcessor
)


class DatasetReader:
    """
    Dataset fabric
    """

    def __init__(self, filesystem: Union[FileSystem, str] = "local", **filesystem_kwargs):
        """
        Parameters
        ----------
        filesystem: str
            'local' | 's3', type of filesystem to use
        **filesystem_kwargs
            kwargs for corresponding DPF.filesystems.FileSystem class
        """
        if filesystem == "local":
            self.filesystem = LocalFileSystem()
        elif filesystem == "s3":
            self.filesystem = S3FileSystem(**filesystem_kwargs)
        elif isinstance(filesystem, FileSystem):
            self.filesystem = filesystem
        else:
            raise NotImplementedError(f"Unknown filesystem format: {filesystem}")

    @staticmethod
    def get_split_index(path: str) -> str:
        return path.split('/')[-1].split('.')[0]

    def _read_dfs(
        self,
        datafiles: List[str],
        validate_columns: bool = True,
        processes: int = 1,
        progress_bar: bool = False,
    ) -> List[Tuple[str, pd.DataFrame]]:
        if len(datafiles) == 0:
            raise ValueError("No datafiles in this path")

        paths_dataframes = thread_map(
            lambda x: (x, self.filesystem.read_dataframe(x)),
            datafiles,
            max_workers=processes,
            disable=not progress_bar
        )

        if validate_columns:
            column_set = set(paths_dataframes[0][1].columns.tolist())
            for path, df in paths_dataframes:
                assert set(df.columns.tolist()) == column_set, (
                    f"Dataframe {path} have different columns. "
                    f"Expected {column_set}, got {set(df.columns)}"
                )
        return paths_dataframes

    def _validate_dfs(
        self,
        config: DatasetConfig,
        paths_dataframes: List[Tuple[str, pd.DataFrame]]
    ):
        required_columns = list(config.columns_mapping.keys())

        column_set = set(paths_dataframes[0][1].columns.tolist())
        for path, df in paths_dataframes:
            df_columns = set(df.columns.tolist())

            for col in required_columns:
                assert col in df_columns, f'Expected {path} to have "{col}" column'
            assert df_columns == column_set, (
                f"Dataframe {path} have different columns. "
                f"Expected {column_set}, got {set(df.columns)}"
            )

    def _convert_to_path_columns(
        self,
        split_suffix: str,
        config: ShardedDatasetConfig,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        split_container_path = config.path.rstrip('/')+'/'+df['split_name']+split_suffix+'/'
        cols_to_drop = set()

        for datatype in config.datatypes:
            if isinstance(datatype, ShardedDataType):
                file_name_column = datatype.modality.sharded_file_name_column
                df[datatype.modality.path_column] = split_container_path+df[file_name_column]
                cols_to_drop.add(file_name_column)

        if len(cols_to_drop) > 0:
            df.drop(columns=list(cols_to_drop), inplace=True)
        return df

    def _rearrange_columns(
        self,
        df: pd.DataFrame,
        config: DatasetConfig
    ) -> pd.DataFrame:
        column_index = []
        for datatype in config.datatypes:
            if datatype.modality.can_be_file:
                column_index.append(datatype.modality.path_column)
                column_index.append(datatype.modality.sharded_file_name_column)
            if datatype.modality.can_be_column:
                column_index.append(datatype.modality.column)

        columns = [i for i in column_index if i in df.columns]
        orig_columns = [i for i in df.columns if i not in columns]
        columns.extend(list(orig_columns))
        return df[columns]

    def _postprocess_dfs(
        self,
        split_suffix: str,
        config: DatasetConfig,
        paths_dataframes: List[Tuple[str, pd.DataFrame]]
    ) -> pd.DataFrame:
        for path, df in paths_dataframes:
            index = self.get_split_index(path)
            df.insert(loc=1, column='split_name', value=index)

        df = pd.concat([d[1] for d in paths_dataframes], ignore_index=True)

        column_mapping = {}
        for k, v in config.columns_mapping.items():
            if k != v:
                column_mapping[k] = v

        if len(column_mapping) > 0:
            df.rename(columns=column_mapping, inplace=True)

        df = self._convert_to_path_columns(split_suffix, config, df)
        df = self._rearrange_columns(df, config)
        return df

    def from_shards(
        self,
        config: ShardsDatasetConfig,
        validate_columns: bool = True,
        processes: int = 1,
        progress_bar: bool = True,
    ) -> ShardsDatasetProcessor:
        dataset_path = config.path.rstrip("/")
        datafiles_ext = config.datafiles_ext.lstrip(".")
        datafiles_ext_dot = '.'+datafiles_ext
        archive_ext = config.archives_ext.lstrip(".")
        archive_ext_dot = '.' + datafiles_ext

        filepaths = self.filesystem.listdir(dataset_path)
        datafiles = [p for p in filepaths if p.endswith(datafiles_ext_dot)]
        archive_paths = [p for p in filepaths if p.endswith(archive_ext_dot)]
        if len(datafiles) == 0:
            raise ValueError("No datafiles in this path")

        # validate paths
        table_paths_set = set(datafiles)
        archive_paths_set = set(archive_paths)
        for filepath in table_paths_set:
            assert filepath.replace(datafiles_ext_dot, archive_ext_dot) in archive_paths_set, \
                f"File {filepath} has not associated archive"
        for filepath in archive_paths_set:
            assert filepath.replace(archive_ext_dot, datafiles_ext_dot) in table_paths_set, \
                f"Archive {filepath} has not associated data file"
        #

        paths_dataframes = self._read_dfs(
            datafiles, validate_columns, processes, progress_bar,
        )
        if validate_columns:
            self._validate_dfs(config, paths_dataframes)

        df = self._postprocess_dfs('.'+archive_ext, config, paths_dataframes)
        processor = ShardsDatasetProcessor(
            filesystem=self.filesystem,
            df=df,
            config=config
        )
        return processor

    def from_files(
        self,
        config: ShardedFilesDatasetConfig,
        validate_columns: bool = True,
        processes: int = 1,
        progress_bar: bool = True,
    ) -> ShardedDatasetProcessor:
        dataset_path = config.path.rstrip("/")
        datafiles_ext = config.datafiles_ext.lstrip(".")

        filepaths = self.filesystem.listdir(dataset_path)
        datafiles = [p for p in filepaths if p.endswith('.'+datafiles_ext)]
        if len(datafiles) == 0:
            raise ValueError("No datafiles in this path")

        # validate paths
        table_paths_set = set(datafiles)
        for filepath in table_paths_set:
            assert filepath.replace('.'+datafiles_ext, '') in filepaths, \
                f"File {filepath} has not associated folder"
        #

        paths_dataframes = self._read_dfs(
            datafiles, validate_columns, processes, progress_bar,
        )
        if validate_columns:
            self._validate_dfs(config, paths_dataframes)

        df = self._postprocess_dfs('', config, paths_dataframes)
        processor = ShardedFilesDatasetProcessor(
            filesystem=self.filesystem,
            df=df,
            config=config
        )
        return processor

    def from_config(
        self,
        config: DatasetConfig,
        **kwargs
    ) -> DatasetProcessor:
        if isinstance(config, ShardsDatasetConfig):
            processor = self.from_shards(config, **kwargs)
        elif isinstance(config, ShardedFilesDatasetConfig):
            processor = self.from_files(config, **kwargs)
        else:
            raise ValueError(f"Unsupported config: {config}")
        return processor

