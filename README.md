# DataProcessingFramework

Фреймворк для работы с датасетами
  
## Contents

- [Installation](#installation)
- [Overview](#overview)
- [Basic usage](#basic-usage)

## Installation

```bash
pip install git+https://github.com/ai-forever/DataProcessingFramework
```
Or you can install from sources:
```bash
git clone https://github.com/ai-forever/DataProcessingFramework
cd DataProcessingFramework
pip install -r requirements.txt
```

## Overview

The framework supports next operations:
1. Reading a dataset
2. Filtering datasets with variety of filters
3. Converting datasets to other formats
4. Validating datasets

## Supported modalities

- Texts
- Images
- Videos

Also framework supports any combination of the modalities list above. For example, text-image or text-video datasets.

## Supported data formats

- Files
- Shards
- ShardedFiles

### Files format

Simply a csv table with paths to image, videos and other columns.

Reading a dataset from _files_ format:
```python
from DPF.configs import FilesDatasetConfig
from DPF.dataset_reader import DatasetReader

config = FilesDatasetConfig.from_modalities(
    'tests/datasets/files_correct/data.csv',
    image_path_col='image_path',
    caption_col='caption'
)

reader = DatasetReader()
processor = reader.from_config(config)
```

### Shards format

In this format, the dataset is divided into shards of N samples each. 
The files of each shard are collected in a tar archive, and the metainformation is collected in a csv file. 
The tar archive and the csv file of the corresponding shard must have the same file names (shard index).

Structure example: 
```
0.tar
0.csv
1.tar
1.csv
...
```

Contents of a `0.csv`:
```csv
image_name, caption
0.jpg, caption for image 1
1.jpg, caption for image 2
...
```

Reading a dataset from _shards_ format:
```python
from DPF.configs import ShardsDatasetConfig
from DPF.dataset_reader import DatasetReader

config = ShardsDatasetConfig.from_modalities(
    'tests/datasets/shards_correct',
    image_name_col='image_name',
    caption_col='caption'
)

reader = DatasetReader()
processor = reader.from_config(config)
```

### Sharded files format

This format is similar to the _shards_, but instead of tar archives, the files are simply stored in folders.

Structure example: 
```
.
├── 0/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── 0.csv
├── 1/
│   ├── 1000.jpg
│   ├── 1001.jpg
│   └── ...
├── 1.csv
└── ...
```

Reading a dataset from _sharded files_ format:
```python
from DPF.configs import ShardedFilesDatasetConfig
from DPF.dataset_reader import DatasetReader

config = ShardedFilesDatasetConfig.from_modalities(
    'tests/datasets/shards_correct',
    image_name_col='image_name',
    caption_col='caption'
)

reader = DatasetReader()
processor = reader.from_config(config)
```

## Basic usage

Reading a dataset:
```python
from DPF.configs import ShardsDatasetConfig
from DPF.dataset_reader import DatasetReader

config = ShardsDatasetConfig.from_modalities(
    'examples/example_dataset/',
    image_name_col='image_name',
    caption_col='caption'
)

reader = DatasetReader()
processor = reader.from_config(config)
```

Read a video dataset:
```python
config = ShardsDatasetConfig.from_modalities(
    'examples/example_video_dataset/',
    video_name_col='image_name',
    caption_col='caption'
)
```

Applying a filter:
```python
from DPF.filters.images.base_images_info_filter import ImageInfoGatherer
datafilter = ImageInfoGatherer(workers=8)
processor.apply_data_filter(datafilter)
processor.df # new columns ['width', 'height', 'is_correct'] are added
```

Converting to other formats:
```python
processor.to_shards(
  'destination/dir/',
  filenaming="counter", # or "uuid"
  keys_mapping={"text": "caption"},
  workers=4
)
```

```python
processor.to_sharded_files(
  'destination/dir/',
  filenaming="counter", # or "uuid"
  keys_mapping={"text": "caption"},
  workers=4
)
```