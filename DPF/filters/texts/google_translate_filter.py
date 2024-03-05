import time
from typing import Dict, List

import numpy as np
import pandas as pd
from deep_translator import GoogleTranslator
from tqdm import tqdm

from DPF.filters import ColumnFilter


def split_on_batches(text_list: List[str], max_symbols: int = 3000) -> List[List[str]]:
    batches = []
    count = 0
    batch = []
    for _, text in enumerate(text_list):
        if len(text) >= max_symbols:
            if len(batch) > 0:
                batches.append(batch)
            batches.append([text])
            batch = []
            count = 0

        count += len(text)
        if count >= max_symbols - len(batch):
            batches.append(batch)
            count = len(text)
            batch = [text]
        else:
            batch.append(text)

    if len(batch) > 0:
        batches.append(batch)
    return batches


def translate_batch(translator, batch: List[str], delimiter: str = '\n\n') -> Dict[str, str]:
    res = translator.translate(delimiter.join(batch)).split(delimiter)
    assert len(batch) == len(res)
    return {batch[i]: res[i] for i in range(len(res))}


class GoogleTranslateFilter(ColumnFilter):
    """
    GoogleTranslateFilter class
    """

    def __init__(
        self,
        text_column_name: str = "text",
        source_lang: str = "auto",
        target_lang: str = "en",
        max_symbols_in_batch: int = 3000,
        timeout: float = 3,
        num_retries_per_batch: int = 1,
        pbar: bool = True
    ):
        super().__init__(1, pbar)
        self.text_column_name = text_column_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_symbols_in_batch = max_symbols_in_batch
        self.timeout = timeout
        self.num_retries_per_batch = num_retries_per_batch
        self.translator = GoogleTranslator(source=source_lang, target=target_lang)

    @property
    def columns_to_process(self) -> List[str]:
        return [self.text_column_name]

    @property
    def schema(self) -> List[str]:
        return [f"{self.text_column_name}_translated"]

    def process(self, row: dict) -> tuple:
        pass

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        texts_to_translate = list(set(df[self.columns_to_process[0]].tolist()))
        batches = split_on_batches(
            list(texts_to_translate),
            self.max_symbols_in_batch
        )

        results = {}
        for _, batch in enumerate(tqdm(batches, disable=not self.pbar)):
            for num_retry in range(self.num_retries_per_batch+1):
                try:
                    results.update(translate_batch(self.translator, batch))
                    break
                except Exception as err:
                    print(f'[{self.__class__.__name__}] {err}, retry: {num_retry}/{self.num_retries_per_batch}')
                    time.sleep(self.timeout)

        res = np.array(list(df[self.columns_to_process[0]].apply(lambda x: results.get(x, None))))
        return res
