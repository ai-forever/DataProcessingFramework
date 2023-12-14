import sys
sys.path.append('../')
sys.path.append('./')

import os
from tqdm import tqdm
import torch
from accelerate import Accelerator

from DPF.configs import ShardsDatasetConfig
from DPF.dataset_reader import DatasetReader
from DPF.filters.images.llava_captioning_filter import LLaVaCaptioningFilter


SAVE_RESULTS_DIR = 'multigpu_filter_res/'
SHARDS_DIR = 'examples/example_dataset/'

config = ShardsDatasetConfig.from_modalities(
    SHARDS_DIR,
    image_name_col='image_name',
    caption_col='caption'
)

reader = DatasetReader()
processor = reader.from_config(config)

accelerator = Accelerator()
device = accelerator.device
print('process', accelerator.process_index)
print('world_size', accelerator.num_processes)

if not os.path.exists(SAVE_RESULTS_DIR):
    os.mkdir(SAVE_RESULTS_DIR)

N = len(processor.df)
start_id = (N // accelerator.num_processes) * accelerator.process_index
end_id = (N // accelerator.num_processes) * (accelerator.process_index + 1)

processor._df = processor._df[start_id:end_id]

datafilter = LLaVaCaptioningFilter(workers=8, prompt='pixart', batch_size=16)
processor.apply_data_filter(datafilter)

processor.df.to_csv(os.path.join(SAVE_RESULTS_DIR, f'{process}.csv'), index=False)