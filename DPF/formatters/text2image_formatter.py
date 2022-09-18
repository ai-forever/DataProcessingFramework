import pandas as pd
import os
import glob
from tqdm import tqdm

from DPF.utils.utils import get_file_extension


ALLOWED_IMAGES_FORMATS = {'jpg', 'jpeg', 'png', 'bmp', 'mpo', 'ppm', 'tiff', 'gif', 'webp'}
DATASETS_FORMATS = {'raw', 'shards', 'image_only'}


class T2IFormatter:
    
    def _read_dataframe(self, filepath: str, filetype: str, **kwargs) -> pd.DataFrame:
        if filetype == 'csv':
            return pd.read_csv(filepath, **kwargs)
        elif filetype == 'parquet':
            return pd.read_parquet(filepath, **kwargs)
        else:
            raise NotImplementedError(f"Unknown file format: {filetype}")
    
    def _normalize_dataframe_cols(self, df: pd.DataFrame):
        columns = ['image_name', 'image_path', 'table_path', 'data_format']
        columns = [i for i in columns if i in df.columns]
        orig_columns = [i for i in df.columns if i not in columns]
        columns.extend(list(orig_columns))
        return df[columns]
    
    def from_webdataset(
        self
    ) -> pd.DataFrame:
        ### TODO
        raise NotImplementedError()
    
    def from_shards(
        self,
        dataset_path: str,
        datafiles_ext: str = 'csv', 
        imagename_colname: str = 'image_name',
        use_abs_path: bool = False,
        progress_bar: bool = False
    ) -> pd.DataFrame:
        ### TODO
        raise NotImplementedError()
    
    def from_raw(
        self,
        dataset_path: str, 
        datafiles_ext: str = 'csv', 
        imagename_colname: str = 'image_name',
        image_ext: str = None,
        use_abs_path: bool = False,
        progress_bar: bool = False
    ) -> pd.DataFrame:
        
        dataset_path = dataset_path.rstrip('/')
        if use_abs_path:
            dataset_path = os.path.abspath(dataset_path)
        datafiles_ext = datafiles_ext.lstrip('.')
        
        datafiles = glob.glob(f'{dataset_path}/*.{datafiles_ext}')
        
        dataframes = []
        for datafile in tqdm(datafiles, disable=not progress_bar):
            df = self._read_dataframe(datafile, datafiles_ext)
            #
            df['table_path'] = datafile
            #
            df['image_name'] = df[imagename_colname]
            if image_ext:
                image_ext = image_ext.lstrip('.')
                df['image_name'] += '.'+image_ext
            #
            df['image_path'] = df['table_path'].str.slice(0,-(len(datafiles_ext)+1))+'/'+df['image_name']
            dataframes.append(df)
        
        df = pd.concat(dataframes, ignore_index=True)
        df['data_format'] = 'raw'
        df = self._normalize_dataframe_cols(df)
        return df
    
    def from_images_in_folder(
        self,
        dirpath: str,
        allowed_image_formats: set = ALLOWED_IMAGES_FORMATS,
        use_abs_path: bool = False,
        progress_bar: bool = False
    ) -> pd.DataFrame:
        
        dirpath = dirpath.rstrip('/')
        if use_abs_path:
            dirpath = os.path.abspath(dirpath)
        
        image_paths = []
        image_names = []
        pbar = tqdm(disable=not progress_bar)
        for root, dirs, files in os.walk(dirpath):
            for filename in files:
                pbar.update(1)
                file_ext = get_file_extension(filename)[1:]
                if file_ext in allowed_image_formats:
                    path = os.path.join(root, filename)
                    image_paths.append(path)
                    image_names.append(filename)
                    
        df = pd.DataFrame({'image_path': image_paths, 'image_name': image_names})
        df['data_format'] = 'image_only'
        return df
    
    def from_image_paths(
        self,
        image_paths: list,
        allowed_image_formats: set = ALLOWED_IMAGES_FORMATS,
        use_abs_path: bool = False,
    ) -> pd.DataFrame:
        ### TODO
        raise NotImplementedError()
        
        image_paths_filtered = []
        for path in image_paths:
            pass
                    
        df = pd.DataFrame({'image_path': image_paths, 'image_name': image_names})
        df['data_format'] = 'image_only'
        return df