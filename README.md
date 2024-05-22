# DataProcessingFramework

**DPF** - a framework for processing and filtering multimodal datasets.

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

Extra requirements: `filters`, `dev`, `llava`, `video_llava`, `lita`

To install extra requirements run: `pip install .[filters]`

## Overview

Framework supports following features:
1. Reading datasets
2. Filtering datasets and calculating metrics using different models and algorithms. Full list of filters can be found [there](docs/filters.md)
3. Effectively transforming data such as videos and images
4. Data filtering and transformation pipelines
5. Converting datasets to other [formats](docs/formats.md)
6. Validating datasets
7. Support for various file systems (local, s3)

DPF allows you to easily filter datasets and add new metadata. You can use various filters and transformations on your data, create pipelines from them and run them efficiently and quickly. Basic code examples for filtering data are given below:

### Basic example
Check out [basic usage](#basic-usage) for more info about DPF's API.

This is a simple example for image deduplication and image aesthetic quality prediction. All filters in DPF extract attributes from the dataset's data and write them into metadata. You can then use these attributes to filter the data according to your needs.

```python
from DPF import ShardsDatasetConfig, DatasetReader

# creating config for dataset
config = ShardsDatasetConfig.from_path_and_columns(
    'examples/example_dataset',
    image_name_col='image_name',
    text_col="caption"
)

# reading dataset's metadata
reader = DatasetReader()
processor = reader.read_from_config(config)

from DPF.filters.images.hash_filters import PHashFilter
datafilter = PHashFilter(sim_hash_size=8, workers=16)  # creating PHash filter
# calculating PHash
# new column "image_phash_8" will be added
processor.apply_data_filter(datafilter)

print('Dataset length before deduplication:', len(processor))
processor.filter_df(~processor.df['image_phash_8'].duplicated())
print('Dataset length after deduplication:', len(processor))

from DPF.filters.images.aesthetic_improved_filter import ImprovedAestheticFilter
datafilter = ImprovedAestheticFilter(
    weights_folder='../weights',  # path to weights folder, will be downloaded to this folder
    device='cuda:0',
    workers=16
)
processor.apply_data_filter(datafilter)

print(processor.df) # printing new dataset's metadata
```

Run [simple_example.py](simple_example.py) file:
```bash
python simple_example.py
```

### Synthetic captions example
Code below generates synthetic captions for images in [shards](docs/formats.md) on remote S3-compatible storage and updates dataset's metadata without downloading shards:

Before running the example below, install extra requirements: `pip install DPF[filters,llava]`

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
print(datafilter.result_columns) # prints list of columns that will be added
# applying filter to dataset
processor.apply_data_filter(datafilter) # new metadata is created

new_column_name = datafilter.result_columns[1] # name of new added column with generated caption

print(processor.df[new_column_name]) # prints generated image captions
# adding new metadata to remote dataset
processor.update_columns([new_column_name], workers=16)
```

You can find more examples [there](examples/)

### Supported data modalities

The framework supports data that has any combination of the following modalities:
- Text
- Image
- Video

> Datasets with several data of the same modality in one sample are not supported.
For example, datasets with following modalities are supported: text-video, text-image, image-video, images, etc. Modalities that are not supported: image2image, image-text-image, etc.

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
  'examples/example_dataset',  # path to shards
  image_name_col='image_name',  # name of column in csv file with image names 
  text_col='caption'  # name of column in csv file with text/captions
)
```

### Reading a dataset
You can read dataset using `DatasetReader.from_config` method:

```python
from DPF import ShardsDatasetConfig, DatasetReader

config = ShardsDatasetConfig.from_path_and_columns(
  'examples/example_dataset',
  image_name_col='image_name',
  text_col='caption'
)

reader = DatasetReader()
processor = reader.read_from_config(config)
```
Example for _sharded files_ format:

```python
from DPF import ShardedFilesDatasetConfig, DatasetReader

config = ShardedFilesDatasetConfig.from_path_and_columns(
  'examples/example_video_dataset',
  video_name_col='video_name',
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

### Pipelines

Pipelines help to combine several filters into one pipeline and process the dataset using it. For example:
```python
from DPF.configs import ShardsDatasetConfig
from DPF.dataset_reader import DatasetReader
from DPF.pipelines import FilterPipeline
from DPF.filters.images.info_filter import ImageInfoFilter
from DPF.filters.images.hash_filters import PHashFilter

reader = DatasetReader()
config = ShardsDatasetConfig.from_path_and_columns(
    "examples/example_dataset",
    image_name_col='image_name',
)
processor = reader.read_from_config(config, workers=4)

pipeline = FilterPipeline("pipeline_example")
pipeline.add_datafilter(
    ImageInfoFilter,
    {'workers': 4},
    processor_run_kwargs={'return_none_on_error': True},
)
pipeline.add_datafilter(PHashFilter, {'workers': 4})
pipeline.add_deduplication(["image_phash_8"])
pipeline.add_shuffle()
pipeline.run(processor)
```

[More about pipelines](docs/pipelines.md)