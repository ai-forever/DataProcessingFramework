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
2. Filter datasets with variety of filters
3. Convert datasets to other formats
4. Validate datasets

## Supported modalities and data formats

Modalities:
- Texts
- Images
- Videos

Data formats:
- Shards
  - ShardedFiles

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
processor.df
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