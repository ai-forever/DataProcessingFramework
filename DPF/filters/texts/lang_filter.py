from py3langid.langid import LanguageIdentifier, MODEL_FILE

from DPF.filters import ColumnFilter


class LangFilter(ColumnFilter):
    """
    LangFilter class
    """

    def __init__(self, text_column_name: str = "caption", workers: int = 16, pbar: bool = True):
        super().__init__(workers, pbar)
        self.lang_identifier = LanguageIdentifier.from_pickled_model(
            MODEL_FILE, norm_probs=True
        )
        self.text_column_name = text_column_name
        self.df_columns = [self.text_column_name]
        self.schema = ["lang", "lang_score"]

    def process(self, row: dict) -> tuple:
        lg, score = self.lang_identifier.classify(row[self.text_column_name])
        return lg, round(score, 2)
