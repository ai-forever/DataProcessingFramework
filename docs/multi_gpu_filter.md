# Multi GPU filter

`MultiGPUDataFilter` is used to run a datafilter on several GPUs.

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

To run a complex datafilter or if you want to manually create a datafilter class use:

```python
from DPF.filters.images.llava_captioning_filter import LLaVaCaptioningFilter
from DPF.filters.multigpu_filter import MultiGPUDataFilter

def init_fn(pbar_pos: int, device: str, params: dict):
    print('INIT FN', pbar_pos, device, params)

    return LLaVaCaptioningFilter(
        workers=8, prompt=params['prompt'], batch_size=16, 
        device=device, _pbar_position=pbar_pos
    )

multigpufilter = MultiGPUDataFilter(
    ['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'],
    datafilter_init_fn=init_fn,
    datafilter_init_fn_kwargs={'prompt': 'short'}
)
processor.apply_multi_gpu_data_filter(multigpufilter)
```

`init_fn` takes two arguments:
- `pbar_pos: int` - progress bar position, it should be passed to `_pbar_position` datafilter arg
- `device: str` - device where filter should run

## More examples

Simple example with complex filter on multi gpu

```python
from DPF.filters.images.hash_filters import PHashFilter
from DPF.filters.images.info_filter import ImageInfoFilter
from DPF.filters import ComplexDataFilter
from DPF.filters.multigpu_filter import MultiGPUDataFilter


def init_fn(pbar_pos: int, device: str):
    print('INIT FN', pbar_pos, device)
    phashfilter = PHashFilter(workers=1)
    infofilter = ImageInfoFilter(workers=1)

    return ComplexDataFilter([phashfilter, infofilter], workers=2, _pbar_position=pbar_pos)


multigpu_datafilter = MultiGPUDataFilter(
    devices=['cuda:0', 'cuda:1'], 
    datafilter_init_fn=init_fn
)
processor.apply_multi_gpu_data_filter(multigpu_datafilter)
```