import pandas as pd
import os
import glob
from tqdm import tqdm


def validate_caption(df):
    df['bad_caption'] = False
    condition = df['caption'].isna() | df['caption'].str.strip().str.len() == 0
    df.loc[condition, 'status'] = False
    df.loc[condition, 'bad_caption'] = True
    return df


def validate_raw(dataset_path):
    pass