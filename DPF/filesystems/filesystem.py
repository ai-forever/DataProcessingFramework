import pandas as pd
import os
import io
from typing import Union, List, Optional

class FileSystem:
    
    def read_file(self, filepath: str, binary: bool) -> io.BytesIO:
        raise NotImplementedError()
        
    def save_file(self, data: Union[str, bytes, io.BytesIO], filepath: str, binary: bool) -> None:
        raise NotImplementedError()
    
    def read_tar(self, filepath: str, **kwargs):
        raise NotImplementedError()
    
    def read_dataframe(self, filepath: str, **kwargs) -> pd.DataFrame:
        filetype = os.path.splitext(filepath)[1] # get extinsion
        filetype = filetype.lstrip('.')
        data = self.read_file(filepath, binary=True)
        if filetype == 'csv':
            return pd.read_csv(data, **kwargs)
        elif filetype == 'parquet':
            return pd.read_parquet(data, **kwargs)
        else:
            raise NotImplementedError(f"Unknown file format: {filetype}")

    def save_dataframe(self, df: pd.DataFrame, filepath: str, **kwargs) -> None:
        filetype = os.path.splitext(filepath)[1] # get extinsion
        filetype = filetype.lstrip('.')
        data = io.BytesIO()
        if filetype == 'csv':
            df.to_csv(data, **kwargs)
        elif filetype == 'parquet':
            df.to_parquet(data, **kwargs)
        else:
            raise NotImplementedError(f"Unknown file format: {filetype}")
        self.save_file(data=data, filepath=filepath, binary=True)
        
    def listdir(self, folder_path: str, filenames_only: Optional[bool] = False) -> List[str]:
        raise NotImplementedError()
        
    def listdir_with_ext(self, folder_path: str, ext: str, filenames_only: Optional[bool] = False) -> List[str]:
        ext = '.'+ext.lstrip('.')
        return [f for f in self.listdir(folder_path, filenames_only=filenames_only) if f.endswith(ext)]
    
    def mkdir(self, folder_path: str) -> None:
        raise NotImplementedError()