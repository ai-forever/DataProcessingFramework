from typing import List, Optional
import os
import pandas as pd
import numpy as np
from scipy.fftpack import dct
from py3langid.langid import LanguageIdentifier, MODEL_FILE

from DPF.filters.utils import identical_collate_fn
from .text_filter import TextFilter


class LangFilter(TextFilter):
    
    def __init__(self, caption_name:str='caption'):
        super(LangFilter, self).__init__(caption_name)

        self.result_columns = ['lang', 'lang_score']
        self.lang_identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
          
    def filter_text(self, row):
        lg, score = self.lang_identifier.classify(row[self.caption_name])
        
        return lg, score
