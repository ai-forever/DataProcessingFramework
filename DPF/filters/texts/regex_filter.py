import re
from typing import Any, Optional

from DPF.filters import ColumnFilter


def replace_matches(caption: str, re_compiled: re.Pattern[str], replacement: str) -> str:
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
        regex_replacement_list: Optional[list[tuple[str, str]]] = None,
        text_column_name: str = "text",
        workers: int = 16,
        pbar: bool = True
    ):
        super().__init__(workers, pbar)

        if regex_replacement_list is None:
            regex_replacement_list = []
        self.compiled_regex_replacement_list: list[tuple[re.Pattern[str], str]] = []

        # compiling regexs
        for regex, replacement in regex_replacement_list:
            self.add_regex(regex, replacement)

        self.text_column_name = text_column_name

    @property
    def columns_to_process(self) -> list[str]:
        return [self.text_column_name]

    @property
    def result_columns(self) -> list[str]:
        return ["clean_caption"]

    def add_regex(self, regex: str, replacement: str) -> None:
        self.compiled_regex_replacement_list.append((re.compile(regex), replacement))

    def process_sample(self, sample: dict[str, Any]) -> list[Any]:
        caption = sample[self.text_column_name]

        for re_compiled, replacement in self.compiled_regex_replacement_list:
            caption = replace_matches(caption, re_compiled, replacement)

        return [caption]
