from typing import List, Optional
import os
import pandas as pd
import numpy as np

from .text_filter import TextFilter

try:
    import re2 as re
except ModuleNotFoundError:
    print("Can't import package re2, using re package. It is recommended to use more efficient re2 package.")
    import re


def replace_matches(caption, re_compiled, replacement):
    iterator = reversed(list(re_compiled.finditer(str(caption).lower().strip())))
    for match in iterator:
        pos = list(match.span())
        caption = caption[:pos[0]] + replacement + caption[pos[1]:]
    return caption
    

class RegexFilter(TextFilter):
    
    def __init__(
            self,
            regex_replacement_list: list = [],
            text_column_name: str = 'caption',
            workers: int = 16
        ):
        super(RegexFilter, self).__init__(text_column_name, workers)
        
        self.compiled_regexs = []
        self.compile_regexs(regex_replacement_list)
        
        self.text_column_name = text_column_name
        self.result_columns = 'clean_caption'

    def add_regex(self, regex, replacement):
        self.compiled_regexs.append((re.compile(regex), replacement))
        
    def compile_regexs(self, regex_replacement_list):
        for regex, replacement in regex_replacement_list:
            self.add_regex(regex, replacement)

    def process(self, row):
        caption = row[self.text_column_name]
        
        for re_compiled, replacement in self.compiled_regexs:
            caption = replace_matches(caption, re_compiled, replacement)
            
        return caption
        