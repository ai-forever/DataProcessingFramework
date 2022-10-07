import pandas as pd

class T2IFilter:
    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError(
                f'Implement run_pipeline in {self.__class__.__name__}'
        )
        
    def process_batch(self, batch) -> dict:
        raise NotImplementedError(
                f'Implement run_pipeline in {self.__class__.__name__}'
        )
        
    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.run(df)