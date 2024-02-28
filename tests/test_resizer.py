import pytest
from DPF.transforms import Resizer, ResizerModes


def test_invalid_params():
    with pytest.raises(TypeError):
        resizer = Resizer(ResizerModes.FIXED)
    for mode in [ResizerModes.MIN_SIZE, ResizerModes.MAX_SIZE]:
        with pytest.raises(AssertionError):
            resizer = Resizer(mode)

    with pytest.raises(TypeError):
        resizer = Resizer(ResizerModes.FIXED, size=768)
    with pytest.raises(TypeError):
        resizer = Resizer(ResizerModes.FIXED, fixed_size=768)
    with pytest.raises(AssertionError):
        resizer = Resizer(ResizerModes.FIXED, fixed_size=(768, 'a'))

    for mode in [ResizerModes.MIN_SIZE, ResizerModes.MAX_SIZE]:
        with pytest.raises(AssertionError):
            resizer = Resizer(mode, fixed_size=(768, 768))
        with pytest.raises(AssertionError):
            resizer = Resizer(mode, size='a')


def test_resizer_fixed():
    resizer = Resizer(ResizerModes.FIXED, fixed_size=(768, 768))

    assert resizer.get_new_size(768, 768) == (768, 768)
    assert resizer.get_new_size(0, 0) == (768, 768)
    assert resizer.get_new_size(1024, 768) == (768, 768)
    assert resizer.get_new_size(768, 1024) == (768, 768)
    assert resizer.get_new_size(2048, 1333) == (768, 768)


def test_resizer_min_size():
    resizer = Resizer(ResizerModes.MIN_SIZE, size=512)

    assert resizer.get_new_size(768, 768) == (512, 512)
    assert resizer.get_new_size(1024, 2048) == (512, 1024)
    assert resizer.get_new_size(2048, 1024) == (1024, 512)
    assert resizer.get_new_size(128, 128) == (128, 128)
    assert resizer.get_new_size(128, 512) == (128, 512)


def test_resizer_max_size():
    resizer = Resizer(ResizerModes.MAX_SIZE, size=512)

    assert resizer.get_new_size(768, 768) == (512, 512)
    assert resizer.get_new_size(1024, 2048) == (256, 512)
    assert resizer.get_new_size(2048, 1024) == (512, 256)
    assert resizer.get_new_size(128, 128) == (128, 128)
    assert resizer.get_new_size(128, 512) == (128, 512)


def test_resizer_always_resize():
    resizer = Resizer(ResizerModes.MAX_SIZE, size=512, downscale_only=False)

    assert resizer.get_new_size(768, 768) == (512, 512)
    assert resizer.get_new_size(256, 256) == (512, 512)
    assert resizer.get_new_size(128, 256) == (256, 512)
    assert resizer.get_new_size(256, 128) == (512, 256)

    resizer = Resizer(ResizerModes.MIN_SIZE, size=512, downscale_only=False)

    assert resizer.get_new_size(768, 768) == (512, 512)
    assert resizer.get_new_size(256, 256) == (512, 512)
    assert resizer.get_new_size(128, 256) == (512, 1024)
    assert resizer.get_new_size(256, 128) == (1024, 512)
