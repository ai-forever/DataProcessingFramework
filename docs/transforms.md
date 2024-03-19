## Transforms

You can transform data in dataset with DPF.
For example, resize videos or photos in dataset.
You can use `DPF.transforms` for these tasks.

> Transformations are currently working only for _files_ and _sharded files_ formats and only on local storage

List of implemented transforms:
- [ImageResizeTransforms](../DPF/transforms/image_resize_transforms.py) - transforms that resizes images
- [VideoResizeTransforms](../DPF/transforms/video_resize_transforms.py) - transforms that resizes videos

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