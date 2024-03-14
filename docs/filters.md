## Фильтрация данных

Фильтры - это модели или алгоритмы, которые позволяют посчитать метрики для датасета.
Фильтры обрабатывают данные и добавляют новые колонки с посчитанными метриками.

Список реализованных фильтров:
- `images`:
  - [ImageInfoFilter](../DPF/filters/images/info_filter.py) - получение базовой информации (высота, ширина) об изображениях
  - [PHashFilter](../DPF/filters/images/hash_filters.py) - подсчет PHash изображений
  - [ImprovedAestheticFilter](../DPF/filters/images/aesthetic_improved_filter.py) - модель определения эстетичности изображений
  - [BLIPCaptioningFilter](../DPF/filters/images/blip_captioning_filter.py) - кэпшенинг моделью BLIP
  - [CLIPLabelsFilter](../DPF/filters/images/cliplabels_filter.py) - определение близости изображения с набором текстов с помощью модели CLIP
  - [LLaVaCaptioningFilter](../DPF/filters/images/llava_captioning_filter.py) - кэпшенинг моделью LLaVA-13b
  - [NSFWFilter](../DPF/filters/images/nsfw_filter.py) - детекция NSFW изображений
  - [CRAFTFilter](../DPF/filters/images/text_detection_filter.py) - детекция текста
  - [OCRFilter](../DPF/filters/images/ocr_filter.py) - распознавание текста
  - [WatermarksFilter](../DPF/filters/images/watermarks_filter.py) - детекция водяных знаков на изображении
- `text-image`:
  - [BlipFilter](../DPF/filters/text2image/blip_filter.py) - определение близости между картинкой и кэпшеном моделью BLIP-2
  - [CLIPFilter](../DPF/filters/text2image/clip_filter.py) - определение близости между картинкой и кэпшеном моделью CLIP
  - [RuCLIPFilter](../DPF/filters/text2image/ruclip_filter.py) - определение близости между картинкой и кэпшеном моделью ru-clip
- `texts`:
  - [LangFilter](../DPF/filters/texts/lang_filter.py) - определение языка текста
  - [RegexFilter](../DPF/filters/texts/regex_filter.py) - фильтрация регулярными выражениями
- `videos`:
  - [VideoInfoFilter](../DPF/filters/videos/info_filter.py) - получение информации о видео (ширина, высота, fps, длительность)
  - [ImageFilterAdapter](../DPF/filters/videos/image_filter_adapter.py) - адаптер картиночных фильтров к одному кадру видео

Пример применения фильтра:
```python
from DPF.filters.images.base_images_info_filter import ImageInfoGatherer
datafilter = ImageInfoGatherer(workers=8)
processor.apply_data_filter(datafilter)
processor.df # new columns ['width', 'height', 'is_correct'] are added
```

### Запуск фильтра на нескольких GPU

Пример запуска фильтра на нескольких GPU:

```python
from DPF.filters.images.llava_captioning_filter import LLaVaCaptioningFilter
from DPF.filters.multigpu_filter import MultiGPUDataFilter

multigpufilter = MultiGPUDataFilter(
    ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
    LLaVaCaptioningFilter,
    dict(
        pbar=True, workers=8,
        prompt='short', batch_size=16
    )
)
processor.apply_multi_gpu_data_filter(multigpufilter)
```