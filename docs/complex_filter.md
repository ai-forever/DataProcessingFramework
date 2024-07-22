# complex filter

`ComplexDataFilter` is used for running several datafilters on one run.

```python
from DPF import DatasetReader
from DPF.configs import ShardsDatasetConfig
from DPF.filters.images.hash_filters import PHashFilter
from DPF.filters.images.info_filter import ImageInfoFilter
from DPF.filters import ComplexDataFilter

config = ShardsDatasetConfig.from_path_and_columns(
    'tests/datasets/shards_correct',
    image_name_col="image_name",
    text_col="caption"
)

reader = DatasetReader()
processor = reader.read_from_config(config)

phashfilter = PHashFilter(workers=1)
infofilter = ImageInfoFilter(workers=1)

datafilter = ComplexDataFilter([phashfilter, infofilter], workers=2)
processor.apply_data_filter(datafilter)
```

A complex filter will iterate over the data only 1 time instead of iterating over the data per filter run.