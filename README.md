# DataProcessingFramework

Фреймворк для обработки мультимодальных датасетов.

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
pip install .
```

## О фреймворке

Фреймворк ориентирован на работу с мультимодальными данными и поддерживает следующие возможности:
1. Чтение датасетов
2. Фильтрация датасетов и подсчет метрик с помощью различных моделей
3. Конвертация датасетов в другие форматы хранения
4. Валидация датасетов

### Поддерживаемые виды данных

Фреймворк поддерживает работу с данными, имеющим любую комбинацию следующих модальностей:
- Тексты
- Изображения
- Видео

При этом, не поддерживаются датасеты, имеющие в одном сэмпле несколько данных одной модальности.
Например, поддерживаются датасеты с модальностями: текст-видео, текст-картинка, картинка-видео, изображение и тд.
Но не поддерживаются датасеты с дублирующимися модальностями: картинка-картинка (img2img), картинка-текст-картинка и тд.

### Поддерживаемые форматы данных

Датасет должен храниться в одном из следующих форматов:
- Files
- Shards
- ShardedFiles

[Подробнее про форматы данных](docs/formats.md)

## Базовое использование

### Конфиги
Чтобы считать датасет, нужно сначала создать конфиг, описывающий датасет и тип данных в нем.
Для каждого формата данных нужно использовать соответствующий конфиг. Пример для формата _shards_:

```python
from DPF.configs import ShardsDatasetConfig

config = ShardsDatasetConfig.from_path_and_columns(
  'examples/example_dataset/',  # путь к датасету
  image_name_col='image_name',  # название колонки в csv с названием изображения
  video_name_col='video_name',  # название колонки в csv с названием видео
  text_col='caption'  # название колонки в csv с кэпшенами
)
```

### Чтение датасета
Считать датасет можно с помощью класса `DatasetReader`, передав конфиг в метод `from_config`:

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
Пример чтения датасета с видео в формате _files_:

```python
from DPF.configs import FilesDatasetConfig
from DPF.dataset_reader import DatasetReader

config = FilesDatasetConfig.from_path_and_columns(
  'examples/example_video_dataset/',
  video_path_col='video_name',
  text_col='caption'
)

reader = DatasetReader()
processor = reader.read_from_config(config)
```

[Примеры чтения данных из других форматов](docs/formats.md)

### Просмотр и изменение датасета

Обработчик (processor) датасета дает интерфейс для взаимодействия и изменения данных.

Более подробно про методы обработчика [см. здесь](docs/processor.md)

### Фильтрация

Фильтры - это модели или алгоритмы, которые позволяют посчитать метрики для датасета.
Фильтры обрабатывают данные и добавляют новые колонки с посчитанными метриками.

Подробнее про фильтры [см. здесь](docs/filters.md)

### Трансформации

С помощью DPF можно изменять данные в датасете, например, изменить размер каждого видео или каждого фото.
Для этого используется трансформации `DPF.transforms`.

Подробнее про трансформации [см. здесь](docs/transforms.md)