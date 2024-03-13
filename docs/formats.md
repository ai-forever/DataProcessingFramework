## Поддерживаемые форматы данных

Датасеты должен быть форматирован в одном из следующих форматов:
- Files
- Shards
- ShardedFiles

### Формат files

Формат files это csv файл с метаданными и путями к изображениям, видео и др. CSV файл может выглядеть примерно так:
```csv
image_path,text,width,height
images/1.jpg,caption,512,512
```

Чтение датасета из формата _files_:

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

### Формат shards

В этом формате датасет разделяется на шарды по `N` сэмплов в каждом.
Файлы в каждом шарде лежат в tar архиве, а метаинформация лежит в csv файле.
tar архив и csv файл каждого шарда должны иметь одинаковые имена (индекс шарда).

Пример структуры: 
```
0.tar
0.csv
1.tar
1.csv
...
```

Файл `0.csv`:
```csv
image_name, caption
0.jpg, caption for image 1
1.jpg, caption for image 2
...
```

Чтение датасета из формата _shards_:

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

### Формат sharded files

Этот формат похож на формат _sharsd_, но вместо tar архивов файлы лежат в папках.

Пример структуры: 
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

Чтение датасета из формата _sharded files_:

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