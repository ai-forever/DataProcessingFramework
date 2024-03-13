## Взаимодействие с датасетом через `DatasetProcessor`

Через обработчик датасета можно:
- Обновлять и менять метаданные
- Применять фильтры
- Применять трансформации
- Конвертировать датасет в другие форматы
- Просматривать семплы из датасета

### Пример
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

### Атрибуты
У обработчика датасета есть три основных атрибута:
- `processor.df` - Датафрейм с метаданными о каждом сэмпле датасета
- `processor.connector` - Коннектор к файловому хранилищу, где лежит датасет. Объект типа `processor.connectors.Connector`
- `processor.config` - Конфиг датасета

### Вывести саммари о датасете

```python
processor.print_summary()
```

### Обновление и изменение метаданных

Обновить существующие или добавить новые колонки с метаинформацией:
```python
processor.update_columns(['old_column_to_update', 'new_column'])
```
Переименовать колонки с метаинформацией:
```python
processor.rename_columns({'old_column': 'new_columns'})
```
Удалить колонки с метаинформацией:
```python
processor.delete_columns(['column_to_delete'])
```

### Просмотр сэмплов из датасета

```python
from PIL import Image
import io

modality2bytes, metadata = processor.get_random_sample()

print(metadata['caption'])
Image.open(io.BytesIO(modality2bytes['image']))
```

### Фильтрация

[Документация по фильтрам](filters.md)

### Трансформации

[Документация по трансформациям](transforms.md)

### Конвертация в другие форматы

Конвертация в формат _shards_:

```python
processor.save_to_shards(
  'destination/dir/',
  filenaming="counter",  # or "uuid"
  rename_columns={"text": "caption"},
  workers=4
)
```

Конвертация в формат _sharded files_

```python
processor.save_to_sharded_files(
    'destination/dir/',
    filenaming="counter",  # or "uuid"
    rename_columns={"text": "caption"},
    workers=4
)
```