from py3langid.langid import LanguageIdentifier, MODEL_FILE

from .text_filter import TextFilter


class LangFilter(TextFilter):

    def __init__(
            self,
            text_column_name: str = 'caption',
            workers: int = 16
        ):
        super().__init__(text_column_name)
        self.lang_identifier = LanguageIdentifier.from_pickled_model(MODEL_FILE, norm_probs=True)

        self.text_column_name = text_column_name
        self.schema = ['lang', 'lang_score']

    def process(self, row):
        lg, score = self.lang_identifier.classify(row[self.text_column_name])
        return lg, round(score, 2)
