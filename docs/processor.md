## `DatasetProcessor` guide

Dataset processor supports following features:
- Update and change metadata
- Apply filters
- Apply transformations
- Convert dataset to other formats
- View samples from a dataset

### Example
```python
from DPF.configs import ShardsDatasetConfig
from DPF.dataset_reader import DatasetReader

config = ShardsDatasetConfig.from_path_and_columns(
  'examples/example_dataset/',
  image_name_col='image_name',
  text_col='caption'
)

reader = DatasetReader()
processor = reader.read_from_config(config)
```

### Attributes
Dataset processor have three main attributes:
- `processor.df` - Pandas dataframe with metadata
- `processor.connector` - A connector to filesystem there dataset is located. Object of type `processor.connectors.Connector`
- `processor.config` - Dataset config

### Print summary about dataset

```python
processor.print_summary()
```

### Update and change metadata

Update existing columns or add new columns:
```python
processor.update_columns(['old_column_to_update', 'new_column'])
```
Rename columns:
```python
processor.rename_columns({'old_column': 'new_columns'})
```
Delete columns:
```python
processor.delete_columns(['column_to_delete'])
```

### View samples

```python
from PIL import Image
import io

modality2bytes, metadata = processor.get_random_sample()

print(metadata['caption'])
Image.open(io.BytesIO(modality2bytes['image']))
```

### Filters

[Filters documentation](filters.md)

### Transformation

[Transforms documentation](transforms.md)

### Convert to other formats

Convert to _shards_ format:

```python
processor.save_to_shards(
  'destination/dir/',
  filenaming="counter",  # or "uuid"
  rename_columns={"text": "caption"},
  workers=4
)
```

Convert to _sharded files_ format:

```python
processor.save_to_sharded_files(
    'destination/dir/',
    filenaming="counter",  # or "uuid"
    rename_columns={"text": "caption"},
    workers=4
)
```