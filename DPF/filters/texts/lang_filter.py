from typing import List, Optional
import os
import pandas as pd
import numpy as np
from py3langid.langid import LanguageIdentifier, MODEL_FILE

from DPF.filters.utils import identical_collate_fn
from .text_filter import TextFilter


class LangFilter(TextFilter):
    
    def __init__(
            self, 
            text_column_name: str = 'caption',
            workers: int = 16
        ):
        super(LangFilter, self).__init__(text_column_name)
        self.lang_identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)
        
        self.text_column_name = text_column_name
        self.result_columns = ['lang', 'lang_score']
          
    def process(self, row):
        lg, score = self.lang_identifier.classify(row[self.text_column_name])
        return lg, round(score, 2)
