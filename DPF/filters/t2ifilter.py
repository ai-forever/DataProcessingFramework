import pandas as pd

class T2IFilter:
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
                f'Implement run in {self.__class__.__name__}'
        )
        
    def preprocess(self, img_bytes, data):
        raise NotImplementedError(
                f'Implement preprocess in {self.__class__.__name__}'
        )
        
    def process_batch(self, batch) -> dict:
        raise NotImplementedError(
                f'Implement process_batch in {self.__class__.__name__}'
        )
        
    @staticmethod
    def add_values_from_batch(main_dict, batch_dict):
        for k, v in batch_dict.items():
            main_dict[k].extend(v)
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.run(df)