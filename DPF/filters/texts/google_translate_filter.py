import time
from typing import Any

import pandas as pd
from deep_translator import GoogleTranslator
from deep_translator.base import BaseTranslator
from tqdm import tqdm

from DPF.filters import ColumnFilter


def split_on_batches(text_list: list[str], max_symbols: int = 3000) -> list[list[str]]:
    batches = []
    count = 0
    batch: list[str] = []
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


def translate_batch(translator: BaseTranslator, batch: list[str], delimiter: str = '\n\n') -> dict[str, str]:
    res = translator.translate(delimiter.join(batch)).split(delimiter)
    assert len(batch) == len(res)
    return {batch[i]: res[i] for i in range(len(res))}


class GoogleTranslateFilter(ColumnFilter):
    """
    Filter for translating texts with google translate api

    Parameters
    ----------
    text_column_name: str = "text"
        Name of column with texts
    source_lang: str = "auto"
        Source language to translate from
    target_lang: str = "en"
        Language to translate to
    max_symbols_in_batch: int = 3000
        Maximum symbols in one request to API.
    timeout: float = 1
        Timeout between requests
    timeout_on_error: float = 3
        Timeout between requests if error occured
    num_retries_per_batch: int = 1
        Number of retries of errors occured
    pbar: bool = True
        Whether to use a progress bar
    """

    def __init__(
        self,
        text_column_name: str = "text",
        source_lang: str = "auto",
        target_lang: str = "en",
        max_symbols_in_batch: int = 3000,
        timeout: float = 1,
        timeout_on_error: float = 3,
        num_retries_per_batch: int = 1,
        pbar: bool = True
    ):
        super().__init__(1, pbar)
        self.text_column_name = text_column_name
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_symbols_in_batch = max_symbols_in_batch
        self.timeout = timeout
        self.timeout_on_error = timeout_on_error
        self.num_retries_per_batch = num_retries_per_batch
        self.translator = GoogleTranslator(source=source_lang, target=target_lang)

    @property
    def columns_to_process(self) -> list[str]:
        return [self.text_column_name]

    @property
    def result_columns(self) -> list[str]:
        return [f"{self.text_column_name}_translated"]

    def process_sample(self, sample: dict[str, Any]) -> list[Any]:
        return []

    def __call__(self, df: pd.DataFrame) -> list[list[Any]]:
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
                    time.sleep(self.timeout_on_error)
            time.sleep(self.timeout)

        res = list(df[self.columns_to_process[0]].apply(lambda x: results.get(x, None)))
        return res
