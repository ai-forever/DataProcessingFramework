from typing import Any, Dict, List

import numpy as np


def get_duplicated_elements(values: List[Any]) -> Any:
    a = np.array(values)
    s = np.sort(a, axis=None)
    return s[:-1][s[1:] == s[:-1]]


def add_error_count(error2count: Dict[str, int], error: str) -> None:
    if error in error2count:
        error2count[error] += 1
    else:
        error2count[error] = 1
