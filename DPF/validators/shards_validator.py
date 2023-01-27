from typing import Dict, List
from glob import glob, iglob
import os
from io import BytesIO
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import pandas as pd
import numpy as np
from PIL import Image
import random
import tarfile

from .t2i_validator import T2IValidator
from .utils import get_duplicated_elements, add_error_count
from DPF.filesystems.filesystem import FileSystem

    
class ShardsValidator(T2IValidator):
    def __init__(
        self, 
        filesystem: FileSystem,
        csv_columns: list, 
        image_name_col: str = "image_name",
        validate_captions: bool = True,
        caption_column: str = "caption",
        validate_tars: bool = True,
    ):
        super().__init__(filesystem, caption_column)
        self.csv_columns = csv_columns
        self.csv_columns_set = set(self.csv_columns)
        self.image_name_col = image_name_col
        
        self.validate_captions = validate_captions
        self.validate_tars = validate_tars
        
    def _validate_tar(
            self, 
            csv_path: str, 
            df: pd.DataFrame,
            errors: dict, 
            error2count: Dict[str, int]
        ):
        tar = self.filesystem.read_tar(csv_path.replace('.csv', '.tar'))
        image_names_in_tar = []
        for c, member in enumerate(tar):
            try:
                img = Image.open(BytesIO(tar.extractfile(member.name).read())) # check image is not broken
            except Exception as err:
                errname = err.__class__.__name__
                errors['ok'] = False
                if errname in errors:
                    errors[errname].append(member.name)
                    add_error_count(error2count, errname)
                else:
                    errors[errname] = [member.name]
                    add_error_count(error2count, errname)
            image_names_in_tar.append(member.name)
        tar.close()

        image_names_in_csv = df[self.image_name_col]
        
        if len(image_names_in_csv) != len(image_names_in_tar):
            errname = 'number of images in csv not equal to number of images in tar'
            errors[errname] = True
            error2count[errname] = 1
            errors['ok'] = False
        else:
            errors['total files'] = len(image_names_in_csv)
            
        image_names_in_csv_set = set(image_names_in_csv)
        image_names_in_tar_set = set(image_names_in_tar)
        
        images_in_csv_but_not_in_tar = image_names_in_csv_set.difference(image_names_in_tar_set)
        images_in_tar_but_not_in_csv = image_names_in_tar_set.difference(image_names_in_csv_set)
        if len(images_in_csv_but_not_in_tar) > 0:
            errname = 'images in csv but not in tar'
            errors[errname] = list(images_in_csv_but_not_in_tar)
            error2count[errname] = len(errors[errname])
            errors['ok'] = False
        if len(images_in_tar_but_not_in_csv) > 0:
            errname = 'images in tar but not in csv'
            errors[errname] = list(images_in_tar_but_not_in_csv)
            error2count[errname] = len(errors[errname])
            errors['ok'] = False
            
        duplicated_images_in_tar = np.unique(get_duplicated_elements(image_names_in_tar))
        if len(duplicated_images_in_tar) > 0:
            errname = 'duplicated images in tar'
            errors[errname] = list(duplicated_images_in_tar)
            error2count[errname] = len(errors[errname])
            errors['ok'] = False
            
        return errors, error2count
        
    def validate_shard(self, csv_path: str):
        errors = {"ok": True}
        error2count = {}
        df = self.filesystem.read_dataframe(csv_path)
        
        missed_columns = self.csv_columns_set.difference(set(df.columns))
        if len(missed_columns) > 0: 
            errname = 'missed columns'
            errors[errname] = list(missed_columns)
            error2count[errname] = 1
            errors['ok'] = False
        
        if self.validate_tars:
            errors, error2count = self._validate_tar(csv_path, df, errors, error2count)
        
        image_names_in_csv = df[self.image_name_col]
        duplicated_images_in_csv = np.unique(get_duplicated_elements(image_names_in_csv))
        if len(duplicated_images_in_csv) > 0:
            errname = 'duplicated images in csv'
            errors[errname] = list(duplicated_images_in_csv)
            error2count[errname] = len(errors[errname])
            errors['ok'] = False
            
        if self.validate_captions:
            errors_caption, error2count_caption = self.validate_caption_df(df)
            errors.update(errors_caption)
            error2count.update(error2count_caption)

        return {csv_path.replace('.csv', ''): errors}, error2count

    def check_shards(
            self,
            shards_dir: str, 
            processes: int = 1
        ) -> (dict, dict, bool):

        shards_dir = shards_dir.rstrip('/')
        files_tar = self.filesystem.listdir_with_ext(shards_dir, '.tar')
        files_csv = self.filesystem.listdir_with_ext(shards_dir, '.csv')
        files_csv_renamed = [i.replace('.tar', '.csv') for i in files_tar]

        assert len(files_csv) != 0, "Not found any .csv files"
        assert len(files_tar)==len(files_csv) and set(files_csv) == set(files_csv_renamed), \
                "Every .tar file should have .csv file with same filename"

        datas = process_map(
            self.validate_shard, files_csv, 
            max_workers=processes, chunksize=1
        )
        csv2errors = {}
        error2count_all = {}
        all_ok = True
        for data, error2count in datas:
            for key, errors in data.items():
                csv2errors[key] = errors
                all_ok = all_ok&errors['ok']
                
            for err in set(error2count.keys()).difference(set(error2count_all.keys())):
                error2count_all[err] = 0
            for err, count in error2count.items():
                error2count_all[err] += count

        return csv2errors, error2count_all, all_ok