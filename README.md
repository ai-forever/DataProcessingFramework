# DataProcessingFramework

Фреймворк для работы с датасетами
  
## Contents

- [Установка](#installation)
- [Краткий обзор](#overview)
- [Примеры](#basic-usage)

## Installation

```bash
git clone https://github.com/ai-forever/DataProcessingFramework
cd DataProcessingFramework
pip install -r requirements.txt
```

## Overview

Данный фреймворк дает возможность:
- Считывать и просматривать датасеты как локально, так и на удаленном хранилище (например, S3)
- Применять различные фильтры для текстов и картинок в датасете
- Сохранять и изменять датасет, добавлять в него новую информацию 
- Валидировать датасет, то есть проверять его на соответствие определенному формату хранения

Во фреймворке используется несколько основных и вспомогательных классов, выполняющие определенные задачи.

**Основные абстракции и их функции:**
- **Formatter** (`DPF.formatters`) - Позволяет считать датасет, создает класс `Processor` для данного датасета
- **Processor** (`DPF.processor`) - Основной класс, инкапсулирует в себя всю работу с датасетом: просмотр семплов, изменение и обновление данных и прочее
- **Filter** (`DPF.filters`) - Представляет собой некоторую функцию, применяемую к датасету с целью получить новую информацию, структурировать или обнаружить неподходящие данные
- **Validator** (`DPF.validators`) - Класс, использующийся для проверки датасета на соответствие определенному формату хранению
- **Pipeline** (`DPF.pipelines`) - Объединяет несколько действий в один пайплайн для упрощения обработки датасета

**Вспомогательные классы:**

- **FileSystem** (`DPF.filesystems`) - Абстракция файловой системы (local/S3)
- **Dataloader** (`DPF.dataloaders`) - Подгрузчики данных для каждого формата хранения
- **Writer** (`DPF.processors.writers`) - Класс, реализующий сохранение данных для конкретного формата хранения

### Поддерживаемые форматы

Поддерживаются следующие модальности и форматы хранения:
- *Images* (просто картинки)
- *Text-to-image* (пары картинка-текст)
  - *shards* (tar-архивы с картинками и csv с текстами и дополнительными данными)
  - *raw* (папки с картинками и csv с текстами и дополнительными данными)

# Basic usage

Примеры работы с фреймворком представлены в папке `examples/`

## Text to image

### Считывание датасета

Для считывания датасета используются методы `formatter.from_*` (вместо `*` пишите нужную вам функцию) класса `T2IFormatter`. Метод вернет экземпляр класса `T2IProcessor`, через который осуществляется работа с датасетом.

Пример считывания картиночно-текстового датасета в формате shards с локального диска:
```python
from DPF.formatters.t2i_formatter import T2IFormatter

formatter = T2IFormatter()

processor = formatter.from_shards(
    'path_to_your_shards', 
    imagename_column='image_name',
    caption_column='caption',
    progress_bar=True,
    processes=8
)
```

Пример считывания датасета с S3:
```python
from DPF.formatters.t2i_formatter import T2IFormatter

formatter = T2IFormatter(
    filesystem='s3',
    key='your_access_key',
    secret='your_secret_key',
    endpoint_url='your_endpoint'
)

processor = formatter.from_shards(
    'path_to_your_dataset_on_s3', 
    imagename_column='image_name',
    caption_column='rus_caption',
    progress_bar=True,
    processes=8
)
```

Основная информация о датасете хранится в атрибуте `processor.df`. 

Больше подробностей можно найти в [ноутбуке с примером](https://github.com/ai-forever/DataProcessingFramework/blob/main/examples/explore_text2image_dataset.ipynb)

### Изменение и добавление данных

Датафрейм `processor.df` можно изменять, а также добавлять к нему новые колонки. Для сохранения изменений используйте метод `processor.update_data` Например, добавление колонки "соотношение сторон" можно произвести следующим способом:
```python
processor.df['aspect_ratio'] = processor.df['width']/processor.df['height']
processor.update_data(['aspect_ratio'], processes=8)
```
Основные методы `T2IProcessor` и их назначение перечислены ниже:
- `processor.rename_columns` - переименование колонок
- `processor.delete_columns` - удаление колонок
- `processor.update_data` - добавление новых колонок и обновление измененных колонок<br>
Все перечисленные выше методы изменяют именно файлы датасета, то есть сохраняют изменения в хранилище.
- `processor.get_random_samples` - просмотр случайных примеров из датасета
- `processor.apply_filter` - применение фильтра к датасету

### Фильтрация

Для датасетов модальности *text2image* можно применять также фильтры модальностей *images* и *texts*. Фильтры соответствующих модальностей расположены в папке `DPF/filters/` следующим образом:
- `DPF/filters/text2image`
- `DPF/filters/images`
- `DPF/filters/texts`

Для применения фильтра необходимо сначала создать объект класса `Processor`, а затем вызвать метод `Processor.apply_filter`, передав ему соответсвующий фильтр. В результате работы фильтра в датафрейме `processor.df` добавятся новые колонки. Например, применение фильтра водяных знаков будет выглядить так:
```python
from DPF.filters.images.watermarks_filter import WatermarksFilter

# see more: help(WatermarksFilter)
watermarks_filter = WatermarksFilter(
    'resnext50_32x4d-small',
    weights_folder='your_weights_folder',
    workers=8, batch_size=128
)

processor.apply_filter(watermarks_filter)
processor.df.head()
```

Фильтр, определяющий язык текста:
```python
from DPF.filters.texts.lang_filter import LangFilter

langfilter = LangFilter(
    text_column_name='caption'
)

processor.apply_filter(lfilter)
processor.df.head()
```

Больше про фильтры смотрите [здесь](#filters), а также в [этом ноутбуке](https://github.com/ai-forever/DataProcessingFramework/blob/main/examples/filter_text2image_dataset.ipynb).

### Валидация

Валидация позволяет проверить соответствует ли датасет формату хранения, также проверить его на битые файлы, значения NaN и прочее. Для этого используется класс `T2IValidator`. Валидатор принимает путь к датасету и возвращает: 
1) список ошибок для каждого шарда
2) для каждой ошибки - сколько раз она возникла
3) статус (`False`, если возникла хотя бы одна ошибка, `True` иначе)

Пример валидации датасета в формате *shards*:
```python
from DPF.formatters.t2i_formatter import T2IFormatter
from DPF.validators.text2image.shards_validator import ShardsValidator

path = 'path_to_your_shards'

formatter = T2IFormatter()
processor = formatter.from_shards(
    path, 
    imagename_column='image_name',
    caption_column='caption',
    progress_bar=True,
    processes=16
)

validator = ShardsValidator(
    filesystem=processor.get_filesystem(),
    csv_columns=['image_name', 'caption'], # список колонок, наличие которых необходимо проверить в датасете
    caption_column='caption',
    validate_tars=True
)

results, err2count, all_ok = validator.validate(path, processes=16)
```

## Images

TO-DO

## Texts

TO-DO

# Filters

TO-DO
