## Supported data formats

The dataset should be stored in one of the following formats::
- Files
- Shards
- ShardedFiles

### Files format

The files format is a csv file with metadata and paths to images, videos, etc. A csv file can look like this:
```csv
image_path,text,width,height
images/1.jpg,caption,512,512
```

Reading a dataset in _files_ format:

```python
from DPF.configs import FilesDatasetConfig
from DPF.dataset_reader import DatasetReader

config = FilesDatasetConfig.from_path_and_columns(
    'tests/datasets/files_correct/data.csv',
    image_path_col='image_path',
    text_col='caption'
)

reader = DatasetReader()
processor = reader.read_from_config(config)
```

### Shards format

In this format, the dataset is divided into shards of N samples each. 
The files in each shard stored in `tar archive, and the metadata is stored in csv file. 
The tar archive and csv file of each shard must have the same names (shard index).

Example of _shards_ structure: 
```
0.tar
0.csv
1.tar
1.csv
...
```

`0.csv` file:
```csv
image_name, caption
0.jpg, caption for image 1
1.jpg, caption for image 2
...
```

Reading a dataset in _shards_ format:

```python
from DPF.configs import ShardsDatasetConfig
from DPF.dataset_reader import DatasetReader

config = ShardsDatasetConfig.from_path_and_columns(
  'tests/datasets/shards_correct',
  image_name_col='image_name',
  text_col='caption'
)

reader = DatasetReader()
processor = reader.read_from_config(config)
```

### Sharded files format

This format is similar to _shards_, but instead of tar archives, the files stored in folders.

Example of _sharded files_ structure: 
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

config = ShardedFilesDatasetConfig.from_path_and_columns(
  'tests/datasets/shards_correct',
  image_name_col='image_name',
  text_col='caption'
)

reader = DatasetReader()
processor = reader.read_from_config(config)
```