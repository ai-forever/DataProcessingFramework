import pandas as pd
import numpy as np
import os
from glob import glob
from tqdm import tqdm
from pandarallel import pandarallel

class T2IValidator:
    
    def validate_caption(self, processor):
        df = processor.df
        df['bad_caption'] = False
        condition = df['caption'].isna() | df['caption'].str.strip().str.len() == 0
        df.loc[condition, 'validate_status'] = False
        df.loc[condition, 'bad_caption'] = True
    
    def validate_duplicates(self, processor, col):
        df = processor.df
        df[f'duplicate_{col}'] = False
        condition = df[col].duplicated()
        df.loc[condition, 'validate_status'] = False
        df.loc[condition, f'duplicate_{col}'] = True

    
class RawValidator(T2IValidator):

    def validate_filestructure(self, processor):
        dataset_path = processor.dataset_path
        datafiles_ext = processor.datafiles_ext
        
        folders = set(glob(f'{dataset_path}/*/'))
        files_data = set(glob(f'{dataset_path}/*.{datafiles_ext}'))
        folders_for_table = set([i.replace('.'+datafiles_ext, '/') for i in files_data])
        
        assert len(files_data) != 0, f"Not found any .{datafiles_ext} files"
        
        folders_without_table = folders.difference(folders_for_table)
        tables_without_folders = folders_for_table.difference(folders)
        if len(folders_without_table) == 0 and len(tables_without_folders) == 0:
            return True, {}
        else:
            if len(folders_without_table) > 0:
                print(f'Found {len(folders_without_table)} folders without any table matching. Check filenames matching')
            if len(tables_without_folders) > 0:
                print(f'Found {len(tables_without_folders)} data files without any folder matching. Check filenames matching')
            return False, {
                "folders_without_table": list(folders_without_table),
                "tables_without_folders": list(tables_without_folders),
            }
    
    def validate_as_subset(self, processor, processes=4, mark_duplicated_image_names=False, mark_bad_caption=True, pbar=True):
        assert (processor.df['data_format']=='raw').all(), "All data should be in raw format"
        
        pandarallel.initialize(nb_workers=processes, progress_bar=pbar)
        
        df = processor.df
        dataset_path = processor.dataset_path
        datafiles_ext = processor.datafiles_ext
        
        df['validate_status'] = True
        
        self.validate_duplicates(processor, 'image_path')
        if mark_duplicated_image_names:
            self.validate_duplicates(processor, 'image_name')
        if mark_bad_caption:
            self.validate_caption(processor)
        # check for existing
        df['file_not_exists'] = False
        condition = df['image_path'].parallel_apply(lambda x: not os.path.exists(x))
        df.loc[condition, 'validate_status'] = False
        df.loc[condition, 'file_not_exists'] = True
        return df['validate_status'].all()
        
    def validate(self, processor, processes=4, samples_per_folder=1000, 
                 mark_duplicated_image_names=False, mark_bad_caption=True):
        assert (processor.df['data_format']=='raw').all(), "All data should be in raw format"
        
        df = processor.df
        dataset_path = processor.dataset_path
        datafiles_ext = processor.datafiles_ext
        
        ok, errors = self.validate_filestructure(processor)
        if not ok:
            print('File structure validation failed. Fix errors and try again.')
            return ok, errors
        
        table_to_data = df.groupby('table_path').apply(
            lambda x: set(x['image_name'].values)
        )
        for table_path, files in table_to_data.items():
            folder_path = table_path[:-len(datafiles_ext)-1] + '/'
            os.listdir(folder_path)