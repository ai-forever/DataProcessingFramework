from typing import Optional

import pandas as pd

from DPF.filesystems import FileSystem


def read_and_validate_df(
    filesystem: FileSystem,
    required_columns: Optional[list[str]],
    path: str
) -> tuple[str, pd.DataFrame]:
    df = filesystem.read_dataframe(path)

    if required_columns:
        for col in required_columns:
            assert col in df.columns, f'Expected {path} to have "{col}" column'

    return path, df


def get_path_filename(path: str) -> str:
    return path.split('/')[-1].split('.')[0]
