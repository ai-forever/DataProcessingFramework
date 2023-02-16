import pandas as pd
from DPF.filesystems import FileSystem


class Filter:
    """
    Abstract class for all filters
    """

    def run(self, df: pd.DataFrame, filesystem: FileSystem) -> pd.DataFrame:
        raise NotImplementedError()
        
    def __call__(self, df: pd.DataFrame, filesystem: FileSystem) -> pd.DataFrame:
        return self.run(df, filesystem)