from typing import List, Optional
import os
import pandas as pd
import numpy as np
from scipy.fftpack import dct
import langid

from DPF.filters.utils import identical_collate_fn
from .text_filter import TextFilter

try:
    import re2 as re
except ModuleNotFoundError:
    print("Can't import package re2, using re package. It is recommended to use more efficient re2 package.")
    import re
import string


class RegexFilter(TextFilter):
    def __init__(
         self, 
         caption_name:str='caption',
         all_regexs: list=[],
         is_regex_ru: bool = True):
            
        super(RegexFilter, self).__init__(caption_name)
        self.result_columns = 'clean_caption'
        self.compiled_regexs = []
        self.compiled_all_regex(all_regexs)

    def add_regex(self, regex, replacement):
        self.compiled_regexs.append((re.compile(regex), replacement))
        
    def compiled_all_regex(self, all_regexs):
        for regex, replacement in all_regexs:
            self.add_regex(regex, replacement)
        
    def replaced_match(self, caption, re_compiled, replacement):
        iterator = reversed(list(re_compiled.finditer(str(caption).lower().strip())))
        for match in iterator:
            pos = list(match.span())
            caption = caption[:pos[0]] + replacement + caption[pos[1]:]
        return caption
        
    def clean_caption(self, caption):
        for re_compiled, replacement in self.compiled_regexs:
            caption = self.replaced_match(caption, re_compiled, replacement)
        return caption
        
    def filter_text(self, row):
        caption = self.clean_caption(row[self.caption_name])
        return caption
        