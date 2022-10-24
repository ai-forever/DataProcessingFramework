import pandas as pd

def read_dataframe(filepath: str, filetype: str, **kwargs) -> pd.DataFrame:
    if filetype == 'csv':
        return pd.read_csv(filepath, **kwargs)
    elif filetype == 'parquet':
        return pd.read_parquet(filepath, **kwargs)
    else:
        raise NotImplementedError(f"Unknown file format: {filetype}")
        
def save_dataframe(df: pd.DataFrame, filepath: str, filetype: str, **kwargs) -> None:
    if filetype == 'csv':
        return df.to_csv(filepath, **kwargs)
    elif filetype == 'parquet':
        return df.to_parquet(filepath, **kwargs)
    else:
        raise NotImplementedError(f"Unknown file format: {filetype}")

def get_file_extension(filepath):
    return filepath[filepath.rfind('.'):]