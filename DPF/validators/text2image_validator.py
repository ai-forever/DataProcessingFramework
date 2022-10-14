import pandas as pd
import os
import glob
from tqdm import tqdm

from DPF.utils.utils import get_file_extension


class T2IValidator:
    def validate_files(self, df):
        raise NotImplementedError()
    
    def validate_basic(self, df, data_format, mark_duplicates=True):
        raise NotImplementedError()
        
        
class RawValidator(T2IValidator):
    def __init__(self):
        pass
    
    def validate(self, df):
        assert (df['data_format']=='raw').all(), "All data should be in raw format for RawValidator"
        
        