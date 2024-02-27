# DataProcessingFramework

Фреймворк для работы с датасетами

## Содержание

- [Установка](#installation)
- [О фреймворке](#overview)
- [Базовое использование](#basic-usage)

## Установка

Установить с помощью pip:
```bash
pip install git+https://github.com/ai-forever/DataProcessingFramework
```
Установка из исходников:
```bash
git clone https://github.com/ai-forever/DataProcessingFramework
cd DataProcessingFramework
pip install -r requirements.txt
```

## О фреймворке

Фреймворк ориентирован на работу с мультимодальными данными и поддерживает следующие возможности:
1. Чтение датасетов
2. Фильтрация датасетов с помощью различных моделей
3. Конвертация датасетов в другие форматы хранения
4. Валидация датасетов

## Поддерживаемые модальности

- Тексты
- Изображения
- Видео

Фреймворк поддерживает работу с комбинацией перечисленных выше модальностей, например, текст-изображение или текст-видео.

## Поддерживаемые форматы данных

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

config = FilesDatasetConfig.from_modalities(
    'tests/datasets/files_correct/data.csv',
    image_path_col='image_path',
    caption_col='caption'
)

reader = DatasetReader()
processor = reader.from_config(config)
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

config = ShardsDatasetConfig.from_modalities(
    'tests/datasets/shards_correct',
    image_name_col='image_name',
    caption_col='caption'
)

reader = DatasetReader()
processor = reader.from_config(config)
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

config = ShardedFilesDatasetConfig.from_modalities(
    'tests/datasets/shards_correct',
    image_name_col='image_name',
    caption_col='caption'
)

reader = DatasetReader()
processor = reader.from_config(config)
```

## Базовое использование

#### Конфиги
Чтобы начать работу с датасетом, нужно сначала создать конфиг `DatasetConfig`, описывающий датасет и его модальности.
Для каждого формата данных нужно использовать соответствующий конфиг. Пример для формата _shards_:
```python
from DPF.configs import ShardsDatasetConfig

config = ShardsDatasetConfig.from_modalities(
    'examples/example_dataset/',  # путь к датасету
    image_name_col='image_name',  # название колонки в csv с названием изображения
    video_name_col='video_name',  # название колонки в csv с названием видео
    caption_col='caption'         # название колонки в csv с кэпшенами
)
```

#### Чтение датасета
Считать датасет можно с помощью класса `DatasetReader`, передав конфиг в метод `from_config`:
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
Пример чтения датасета с видео в формате _sharded files_:
```python
from DPF.configs import ShardedFilesDatasetConfig
config = ShardedFilesDatasetConfig.from_modalities(
    'examples/example_video_dataset/',
    video_name_col='image_name',
    caption_col='caption'
)

reader = DatasetReader()
processor = reader.from_config(config)
```
Взаимодействие с датасетом происходит через интерфейс `DatasetProcessor`. 
`DatasetProcessor` дает возможность валидировать, просматривать, фильтровать датасеты.  

#### Просмотр и изменение датасета

Обработчик датасета (интерфейс `DatasetProcessor`) даёт возможность посмотреть статистику и информацию по датасету, а также посмотреть рандомные семплы данных.

Получить датафрейм датасета можно, обратившись к атрибуту `df`:
```python
processor.df
```
Вывести саммари о датасете:
```python
processor.summary()
```
Обновить существующие или добавить новые колонки:
```python
processor.update_columns(['column_to_update', 'new_column'])
```
Переименовать колонки:
```python
processor.rename_columns({'old_column': 'new_columns'})
```
Удалить колонки:
```python
processor.delete_columns(['column_to_delete'])
```
Получить рандомный семпл из датасета:
```python
modality2bytes, metadata = processor.get_random_sample()
```

#### Фильтрация

Фильтры - это модели или алгоритмы, которые позволяют посчитать некоторый набор метрик для датасета.
Фильтры обрабатывают данные и добавляют новые колонки с посчитанными метриками.

Список фильтров:
- `images`:
  - [ImageInfoFilter](DPF/filters/images/info_filter.py) - получение базовой информации (высота, ширина) об изображениях
  - [PHashFilter](DPF/filters/images/hash_filters.py) - подсчет PHash изображений
  - [ImprovedAestheticFilter](DPF/filters/images/aesthetic_improved_filter.py) - модель определения эстетичности изображений
  - [BLIPCaptioningFilter](DPF/filters/images/blip_captioning_filter.py) - кэпшенинг моделью BLIP
  - [CLIPLabelsFilter](DPF/filters/images/cliplabels_filter.py) - определение близости изображения с набором текстов с помощью модели CLIP
  - [LLaVaCaptioningFilter](DPF/filters/images/llava_captioning_filter.py) - кэпшенинг моделью LLaVA-13b
  - [NSFWFilter](DPF/filters/images/nsfw_filter.py) - детекция NSFW изображений
  - [CRAFTFilter](DPF/filters/images/text_detection_filter.py) - детекция текста
  - [OCRFilter](DPF/filters/images/ocr_filter.py) - распознавание текста
  - [WatermarksFilter](DPF/filters/images/watermarks_filter.py) - детекция водяных знаков на изображении
- `text-image`:
  - [BlipFilter](DPF/filters/text2image/blip_filter.py) - определение близости между картинкой и кэпшеном моделью BLIP-2
  - [CLIPFilter](DPF/filters/text2image/clip_filter.py) - определение близости между картинкой и кэпшеном моделью CLIP
  - [RuCLIPFilter](DPF/filters/text2image/ruclip_filter.py) - определение близости между картинкой и кэпшеном моделью ru-clip
- `texts`:
  - [LangFilter](DPF/filters/texts/lang_filter.py) - определение языка текста
  - [RegexFilter](DPF/filters/texts/regex_filter.py) - фильтрация регулярными выражениями
- `videos`:
  - [VideoInfoFilter](DPF/filters/videos/info_filter.py) - получение информации о видео (ширина, высота, fps, длительность)
  - [ImageFilterAdapter](DPF/filters/videos/image_filter_adapter.py) - адаптер картиночных фильтров к одному кадру видео

Применение фильтра:
```python
from DPF.filters.images.base_images_info_filter import ImageInfoGatherer
datafilter = ImageInfoGatherer(workers=8)
processor.apply_data_filter(datafilter)
processor.df # new columns ['width', 'height', 'is_correct'] are added
```

#### Трансформация датасета
С помощью DPF можно изменять данные в датасете, например, изменить размер каждого видео или каждого фото.
Для этого используется трансформации `DPF.transforms`.

Уменьшить все изображения до 768 пикселей по минимальной стороне с сохранением соотношения сторон:
```python
from DPF.transforms import ImageResizeTransforms, Resizer, ResizerModes

transforms = ImageResizeTransforms(Resizer(ResizerModes.MIN_SIZE, size=768))
processor.apply_transform(transforms)
```

Уменьшить все видео до 768 пикселей по максимальной стороне с сохранением соотношения сторон:
```python
from DPF.transforms import VideoResizeTransforms, Resizer, ResizerModes

transforms = VideoResizeTransforms(Resizer(ResizerModes.MAX_SIZE, size=768))
processor.apply_transform(transforms)
```

#### Конвертация между форматами

Конвертация в формат _shards_:
```python
processor.to_shards(
  'destination/dir/',
  filenaming="counter", # or "uuid"
  keys_mapping={"text": "caption"},
  workers=4
)
```

Конвертация в формат _sharded files_
```python
processor.to_sharded_files(
  'destination/dir/',
  filenaming="counter", # or "uuid"
  keys_mapping={"text": "caption"},
  workers=4
)
```