## Filters

Filters are models or algorithms that calculate metrics for a dataset. 
Filters process the data and add new columns with the calculated metrics.

List of implemented filters:
- `images`:
  - [ImageInfoFilter](../DPF/filters/images/info_filter.py) - gather basic info about images (width, height, etc)
  - [PHashFilter](../DPF/filters/images/hash_filters.py) - PHash for images
  - [ImprovedAestheticFilter](../DPF/filters/images/aesthetic_improved_filter.py) - aesthetic scoring for images
  - [BLIPCaptioningFilter](../DPF/filters/images/blip_captioning_filter.py) - captioning images using BLIP model
  - [CLIPLabelsFilter](../DPF/filters/images/cliplabels_filter.py) - calculate similarity of images with provided texts using CLIP model
  - [LLaVaCaptioningFilter](../DPF/filters/images/llava_captioning_filter.py) - captioning images using LLaVA models
  - [NSFWFilter](../DPF/filters/images/nsfw_filter.py) - NSFW images detection
  - [CRAFTFilter](../DPF/filters/images/text_detection_filter.py) - text detection on image
  - [OCRFilter](../DPF/filters/images/ocr_filter.py) - text recognition
  - [WatermarksFilter](../DPF/filters/images/watermarks_filter.py) - watermarks detection
- `text-image`:
  - [BlipFilter](../DPF/filters/text2image/blip_filter.py) - similarity of images and texts using BLIP-2
  - [CLIPFilter](../DPF/filters/text2image/clip_filter.py) - similarity of images and texts using CLIP
  - [RuCLIPFilter](../DPF/filters/text2image/ruclip_filter.py) - similarity of images and texts using ru-clip
- `texts`:
  - [LangFilter](../DPF/filters/texts/lang_filter.py) - text language classification
  - [GoogleTranslateFilter](../DPF/filters/texts/google_translate_filter.py) - translates a text
  - [RegexFilter](../DPF/filters/texts/regex_filter.py) - filter texts using regular expressions
- `videos`:
  - [VideoInfoFilter](../DPF/filters/videos/info_filter.py) - gather basic info about videos (width, height, fps, duration)
  - [ImageFilterAdapter](../DPF/filters/videos/image_filter_adapter.py) - adapter of image filters to the one frame of video
  - [GunnarFarnebackFilter](../DPF/filters/videos/farneback_filter.py) - computes flow scores using Farneback's algorithm
  - [RAFTOpticalFlowFilter](../DPF/filters/videos/raft_filter.py) - computes flow scores using [RAFT](https://github.com/princeton-vl/RAFT) model
  - [VideoLLaVAFilter](../DPF/filters/videos/video_llava_filter.py) - captioning videos using Video-LLaVA

### Datafilter

Datafilters are filters that calculate new metadata (scores, captions, probabilities, etc) based on a file modalities: images and videos.
To run a datafilter, use `processor.apply_data_filter()` method.

Example of using datafilter that adds metadata about images (width, height, channels):
```python
from DPF.filters.images.info_filter import ImageInfoFilter
datafilter = ImageInfoFilter(workers=8)
processor.apply_data_filter(datafilter)
processor.df # new columns ['width', 'height', 'is_correct'] are added
```

### Columnfilter

Columnfilters are filters that also calculates new metadata, but based on a existing metadata (texts, etc).
To run a columnfilter, use `processor.apply_column_filter()` method.

Example of using column filter that classifies the text language:
```python
from DPF.filters.texts.lang_filter import LangFilter

columnfilter = LangFilter(workers=16)
processor.apply_column_filter(columnfilter)
processor.df # new columns ["lang", "lang_score"] are added
```

### Running filter on several GPUs

To run a datafilter on multiple GPUs use `MultiGPUDataFilter` class:

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
See `help(MultiGPUDataFilter)` for more information.

### Examples

You can find usage examples [there](../examples).
- [Image filters examples](../examples/image_filters_example.ipynb)
- [Video filters examples](../examples/video_filters_example.ipynb)
- [Text filters examples](../examples/text_filters_example.ipynb)

### Creating new filter

TODO

