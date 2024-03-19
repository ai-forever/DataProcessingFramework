# DataProcessingFramework

A framework for processing and filtering multimodal datasets.

- [Installation](#installation)
- [Overview](#overview)
- [Basic usage](#basic-usage)

## Installation

Install with pip:
```bash
pip install git+https://github.com/ai-forever/DataProcessingFramework
```
Install from repository:
```bash
git clone https://github.com/ai-forever/DataProcessingFramework
cd DataProcessingFramework
pip install .
```

## Overview

Framework supports following features:
1. Reading datasets
2. Filtering datasets and calculating metrics using different models
3. Converting datasets to other storage formats
4. Datasets validating
5. Supports different filesystems (local, s3)
6. Data filtering pipelines

DPF allows you to easily filter datasets and add new metadata. 
For example, the code below generates synthetic captions for images in shards on remote s3 storage and updates dataset metadata without downloading shards:
```python
from DPF import S3Connector, DatasetReader, ShardsDatasetConfig

# creating connector for S3 storage
connector = S3Connector(
    key='access_key',
    secret='secret_key',
    endpoint_url='endpoint_url'
)

reader = DatasetReader(connector)

# creating dataset config
config = ShardsDatasetConfig.from_path_and_columns(
    "s3://your-bucket/path/to/shards",
    image_name_col='image_name',
)
# reading a dataset
processor = reader.read_from_config(config, workers=16)

from DPF.filters.images.llava_captioning_filter import LLaVaCaptioningFilter

# creating LLaVA captioner filter
datafilter = LLaVaCaptioningFilter(
    workers=16, prompt='short', 
    batch_size=16, device="cuda:0"
)
# applying filter to dataset
processor.apply_data_filter(datafilter) # new metadata is created

print(processor.df[datafilter.schema[1]]) # prints generated image captions
# adding new metadata to remote dataset
processor.update_columns([datafilter.schema[1]], workers=16)
```

### Supported data modalities

The framework supports data that has any combination of the following modalities:
- Text
- Image
- Video

> Datasets with several data of the same modality in one sample are not supported.
For example, datasets with following modalities are supported: text-video, text-image, image-video, images, etc.
Modalities that are not supported: image2image, image-text-image, etc.

### Supported data formats

The dataset should be stored in one of the following formats:
- Files
- Shards
- Sharded files

[More about data formats](docs/formats.md)

## Basic usage

### Configs
To read a dataset, you must first create a config that describes the dataset and the type of data in it.
For each data format, you need to use the appropriate config.

Example for _shards_ format:

```python
from DPF import ShardsDatasetConfig

config = ShardsDatasetConfig.from_path_and_columns(
  'examples/example_dataset/',  # path to shards
  image_name_col='image_name',  # name of column in csv file with image names 
  text_col='caption'  # name of column in csv file with text/captions
)
```

### Reading a dataset
You can read dataset using `DatasetReader.from_config` method:

```python
from DPF import ShardsDatasetConfig, DatasetReader

config = ShardsDatasetConfig.from_path_and_columns(
  'examples/example_dataset/',
  image_name_col='image_name',
  text_col='caption'
)

reader = DatasetReader()
processor = reader.read_from_config(config)
```
Example for _files_ format:

```python
from DPF import FilesDatasetConfig, DatasetReader

config = FilesDatasetConfig.from_path_and_columns(
  'examples/example_video_dataset/',
  video_path_col='video_name',
  text_col='caption'
)

reader = DatasetReader()
processor = reader.read_from_config(config)
```

[Examples of reading data in other formats](docs/formats.md)

Example reading a dataset directly from S3 storage:
```python
from DPF import S3Connector, DatasetReader, ShardsDatasetConfig

connector = S3Connector(
    key='access_key',
    secret='secret_key',
    endpoint_url='endpoint_url'
)
reader = DatasetReader(connector)

config = ShardsDatasetConfig.from_path_and_columns(
    "s3://your-bucket/path/to/shards",
    image_name_col='image_name',
)
processor = reader.read_from_config(config, workers=16)
```

### Viewing and updating dataset

A dataset processor provides an interface for interacting with data and modifying it.

[More about dataset processor](docs/processor.md)

### Filtering dataset

Filters are models or algorithms that calculate metrics for a dataset. 
Filters process the data and add new columns with the calculated metrics.

[More about filters](docs/filters.md)

### Transforming dataset

You can transform data in dataset with DPF.
For example, resize videos or photos in dataset.
You can use `DPF.transforms` for these tasks.

[More about transforms](docs/transforms.md)