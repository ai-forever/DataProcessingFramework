from typing import Any

from py3langid.langid import MODEL_FILE, LanguageIdentifier

from DPF.filters import ColumnFilter


class LangFilter(ColumnFilter):
    """
    Filter for text language detection

    Parameters
    ----------
    text_column_name: str = "text"
        Name of column with texts
    workers: int = 16
        Number of processes to use
    pbar: bool = True
        Whether to use a progress bar
    """

    def __init__(
        self,
        text_column_name: str = "text",
        workers: int = 16,
        pbar: bool = True
    ):
        super().__init__(workers, pbar)
        self.lang_identifier = LanguageIdentifier.from_pickled_model(
            MODEL_FILE, norm_probs=True
        )
        self.text_column_name = text_column_name

    @property
    def columns_to_process(self) -> list[str]:
        return [self.text_column_name]

    @property
    def result_columns(self) -> list[str]:
        return ["lang", "lang_score"]

    def process_sample(self, sample: dict[str, Any]) -> list[Any]:
        lg, score = self.lang_identifier.classify(sample[self.text_column_name])
        return [lg, round(score, 2)]
