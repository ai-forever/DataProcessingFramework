from typing import Optional, List
import re

from DPF.filters import ColumnFilter


def replace_matches(caption, re_compiled, replacement):
    iterator = reversed(list(re_compiled.finditer(str(caption).lower().strip())))
    for match in iterator:
        pos = list(match.span())
        caption = caption[:pos[0]] + replacement + caption[pos[1]:]
    return caption


class RegexFilter(ColumnFilter):
    """
    RegexFilter class
    """

    def __init__(
        self,
        regex_replacement_list: Optional[list] = None,
        text_column_name: str = "text",
        workers: int = 16,
        pbar: bool = True
    ):
        super().__init__(workers, pbar)

        if regex_replacement_list is None:
            regex_replacement_list = []
        self.compiled_regexs = []
        self.compile_regexs(regex_replacement_list)

        self.text_column_name = text_column_name

    @property
    def columns_to_process(self) -> List[str]:
        return [self.text_column_name]

    @property
    def schema(self) -> List[str]:
        return ["clean_caption"]

    def add_regex(self, regex, replacement):
        self.compiled_regexs.append((re.compile(regex), replacement))

    def compile_regexs(self, regex_replacement_list):
        for regex, replacement in regex_replacement_list:
            self.add_regex(regex, replacement)

    def process(self, row: dict) -> tuple:
        caption = row[self.text_column_name]

        for re_compiled, replacement in self.compiled_regexs:
            caption = replace_matches(caption, re_compiled, replacement)

        return caption
