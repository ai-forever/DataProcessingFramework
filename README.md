# DataProcessingFramework

Фреймворк для работы с датасетами

Поддерживаемые форматы:
- Images
- Text-to-image
  - shards
  - raw
  
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

# Basic usage

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

Больше подробностей можно найти в [ноутбуке с примером]()

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
