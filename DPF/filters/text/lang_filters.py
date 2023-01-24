from typing import List, Optional
import os
import pandas as pd
import numpy as np
from scipy.fftpack import dct
import py3langid as langid

from DPF.filters.utils import identical_collate_fn
from .text_filter import TextFilter


# def is_lang(text):
#     lg, score = langid.classify(text)
#     return lg


class LangFilter(TextFilter):
    
    def __init__(self, caption_name:str='caption'):
        super(LangFilter, self).__init__(caption_name)

        self.result_columns = ['lang', 'lang_score']
     
          
    def filter_text(self, row):
        lg, score = langid.classify(row[self.caption_name])
        return lg, score
        
        