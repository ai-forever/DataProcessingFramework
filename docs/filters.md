# Filters

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
  - [LLaVa34bCaptioningFilter](../DPF/filters/images/llava34b_captioning_filter.py) - captioning images using LLaVA models, llava-v1.6-34b-hf
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
  - [LITAFilter](../DPF/filters/videos/lita_filter.py) - captioning videos using [LITA model](https://github.com/NVlabs/LITA)
  - [PllavaFilter](../DPF/filters/videos/pllava_filter.py) - captioning videos using [Pllava model](https://pllava.github.io)
  - [GroundingGPTFilter](../DPF/filters/videos/grounding_gpt_filter.py) - captioning using [grounding gpt model](https://github.com/lzw-lzw/GroundingGPT)
- `audios`:
  - [AudioInfoFilter](../DPF/filters/audios/info_filter.py) - gather basic info about audios (duration, sample_rate, correctness)

## Datafilter

Datafilters are filters that calculate new metadata (scores, captions, probabilities, etc) based on a file modalities: images and videos.
To run a datafilter, use `processor.apply_data_filter()` method.

Example of using datafilter that adds metadata about images (width, height, channels):
```python
from DPF.filters.images.info_filter import ImageInfoFilter
datafilter = ImageInfoFilter(workers=8)
processor.apply_data_filter(datafilter)
processor.df # new columns ['width', 'height', 'is_correct'] are added
```

## Columnfilter

Columnfilters are filters that also calculates new metadata, but based on a existing metadata (texts, etc).
To run a columnfilter, use `processor.apply_column_filter()` method.

Example of using column filter that classifies the text language:
```python
from DPF.filters.texts.lang_filter import LangFilter

columnfilter = LangFilter(workers=16)
processor.apply_column_filter(columnfilter)
processor.df # new columns ["lang", "lang_score"] are added
```

## Examples

You can find usage examples [there](../examples).
- [Image filters examples](../examples/image_filters_example.ipynb)
- [Video filters examples](../examples/video_filters_example.ipynb)
- [Text filters examples](../examples/text_filters_example.ipynb)

## Creating new filter

To add your filter, you should create new filter class.
If your filter uses only data from columns (e.g. _text_ modality), you should inherit your class from [ColumnFilter class](../DPF/filters/column_filter.py)
If your filter uses data from files, you should inherit your class from [DataFilter class](../DPF/filters/data_filter.py)

### Creating DataFilter

To create a new datafilter, add new file in a folder with the modality used by your filter. 
For example, if your filter uses _images_ modality, create file in [DPF/filters/images/](../DPF/filters/images) folder.
If your filter uses _texts_ and _images_ modality, create file in [DPF/filters/text2image/](../DPF/filters/text2image) and so on.

Inherit you filter from corresponding `DataFilter` class in modality folder:
- [DPF/filters/images/img_filter.py](../DPF/filters/images/img_filter.py) for _images_
- [DPF/filters/text2image/t2i_filter.py](../DPF/filters/text2image/t2i_filter.py) for _texts_ and _images_
- [DPF/filters/videos/video_filter.py](../DPF/filters/videos/video_filter.py) for _videos_

Then you should implement `result_columns`, `dataloader_kwargs` properties and `preprocess_data`, `process_batch` methods.
- `result_columns` - list of result columns that filter adds to a DataFrame
- `dataloader_kwargs` - parameters for a pytorch dataloader
- `preprocess_data` - method where data preprocessing is implemented. This method is passed to dataloader and preprocessing runs in multiple processes. Do not use cuda operations in this method.
- `process_batch` - method where batch is processed with model

For more information run:
```python
from DPF.filters import DataFilter
help(DataFilter)
```

Example of custom DataFilter:
```python
from typing import Any

from DPF.filters.images.img_filter import ImageFilter
from DPF.types import ModalityToDataMapping

class PHashFilter(ImageFilter):
    def __init__(
        self,
        sim_hash_size: int = 8,
        workers: int = 16,
        pbar: bool = True,
        _pbar_position: int = 0
    ):
        super().__init__(pbar, _pbar_position)
        self.num_workers = workers
        self.sim_hash_size = sim_hash_size

    @property
    def result_columns(self) -> list[str]:
        return [f"image_phash_{self.sim_hash_size}"]

    @property
    def dataloader_kwargs(self) -> dict[str, Any]:
        return {"num_workers": self.num_workers, "batch_size": 1, "drop_last": False}

    def preprocess_data(
        self,
        modality2data: ModalityToDataMapping,
        metadata: dict[str, Any]
    ) -> Any:
        key = metadata[self.key_column]
        img_simhash = get_phash(
            read_image_rgb_from_bytes(modality2data['image']), 
            hash_size=self.sim_hash_size
        )
        return key, img_simhash

    def process_batch(self, batch: list[Any]) -> dict[str, list[Any]]:
        df_batch_labels = self._get_dict_from_schema()

        keys, img_simhashes = list(zip(*batch))
        df_batch_labels[self.key_column].extend(keys)
        df_batch_labels[f"image_phash_{self.sim_hash_size}"].extend(img_simhashes)

        return df_batch_labels
```

This filter reads images and calculates PHash **in dataloader**. 
Then dataloader returns PHash strings and these strings are added in result dataframe. 

### Creating ColumnFilter

To create a new columnfilter, add new file in a folder with the modality used by your filter.
Inherit your class from [ColumnFilter](../DPF/filters/column_filter.py) class.

Then you should implement `result_columns`, `columns_to_process` properties and `process_sample` methods.
- `result_columns` - list of result columns that filter adds to a DataFrame
- `columns_to_process` - columns in original dataframe used for processing. These columns are being passed in method 
- `process_sample` - method that processes one sample of data.

For more information run:
```python
from DPF.filters import ColumnFilter
help(ColumnFilter)
```

Example of custom ColumnFilter:
```python
from typing import Any
from py3langid.langid import MODEL_FILE, LanguageIdentifier
from DPF.filters import ColumnFilter

class LangFilter(ColumnFilter):
    """
    LangFilter class
    """

    def __init__(
        self,
        text_column_name: str = "text",
        workers: int = 16,
        pbar: bool = True
    ):
        super().__init__(workers, pbar)
        self.lang_identifier = LanguageIdentifier.from_pickled_model(
            MODEL_FILE, norm_probs=True
        )
        self.text_column_name = text_column_name

    @property
    def columns_to_process(self) -> list[str]:
        return [self.text_column_name]

    @property
    def result_columns(self) -> list[str]:
        return ["lang", "lang_score"]

    def process_sample(self, sample: dict[str, Any]) -> list[Any]:
        lg, score = self.lang_identifier.classify(sample[self.text_column_name])
        return [lg, round(score, 2)]
```

This filter creates 2 new columns: `lang` and `lang_score`. 
It uses column with text name to identify the language of a text.