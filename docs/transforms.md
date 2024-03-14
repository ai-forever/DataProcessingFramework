## Трансформации датасета

С помощью DPF можно изменять данные в датасете, например, изменить размер каждого видео или каждого фото.
Для этого используется трансформации `DPF.transforms`.

Трансформации пока что работают только для форматов _files_ и _sharded files_.

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