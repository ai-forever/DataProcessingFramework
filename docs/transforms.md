## Transforms

You can transform data in dataset with DPF.
For example, resize videos or photos in dataset.
You can use `DPF.transforms` for these tasks.

> Transformations are currently working only for _files_ and _sharded files_ formats and only on local storage

List of implemented transforms:
- [ImageResizeTransforms](../DPF/transforms/image_resize_transforms.py) - transforms that resizes images
- [VideoFFMPEGTransforms](../DPF/transforms/video_ffmpeg_transforms.py) - transforms that resizes and changing fps of videos using ffmpeg

### Examples

Resize all images to 768 pixels on the minimum side while maintaining the aspect ratio:
```python
from DPF.transforms import ImageResizeTransforms, Resizer, ResizerModes

transforms = ImageResizeTransforms(Resizer(ResizerModes.MIN_SIZE, size=768))
processor.apply_transform(transforms)
```

Resize all images to 768 pixels on the maximum side while maintaining the aspect ratio:
```python
from DPF.transforms import VideoResizeTransforms, Resizer, ResizerModes

transforms = VideoResizeTransforms(Resizer(ResizerModes.MAX_SIZE, size=768))
processor.apply_transform(transforms)
```

Change fps to 24 and resize all videos to 768 pixels on the minimum side while maintaining the aspect ratio:
```python
from DPF.transforms import VideoFFMPEGTransforms, Resizer, ResizerModes

transforms = VideoFFMPEGTransforms(
    resizer=Resizer(ResizerModes.MIN_SIZE, size=768),
    fps=24,
    workers=8
)
processor.apply_transform(transforms)
```

Fast video cutting:
_cut_start_ and _cut_duration_ columns should have values in seconds. 
For example, cutting video from 7 to 11 second should be specified in dataframe as: _cut_start_ = 7, _cut_duration_ = 4
```python
from DPF.transforms import VideoFFMPEGTransforms

transforms = VideoFFMPEGTransforms(
    cut_start_col='cut_start',
    cut_duration_col='cut_duration',
    copy_stream=True,
    workers=4,
)
processor.apply_transform(transforms)
```