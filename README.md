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

## Basic usage

