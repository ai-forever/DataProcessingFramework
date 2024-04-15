from DPF import DatasetReader
from DPF.configs import ShardsDatasetConfig
from DPF.filters.images.dummy_gpu_filter import DummyGPUFilter
from DPF.filters.images.hash_filters import PHashFilter
from DPF.filters.images.info_filter import ImageInfoFilter
from DPF.pipelines.filter_pipeline import FilterPipeline


def test_pipeline_imageinfo():
    path = 'tests/datasets/shards_correct'
    config = ShardsDatasetConfig.from_path_and_columns(
        path,
        image_name_col="image_name",
        text_col="caption"
    )

    reader = DatasetReader()
    processor = reader.read_from_config(config)

    pipeline = FilterPipeline("test_pipeline_imageinfo")
    pipeline.add_datafilter(
        ImageInfoFilter,
        {'workers': 1},
    )
    pipeline.run(processor)

    assert len({'width', 'height', 'channels'}.intersection(set(processor.df.columns))) == 3


def test_pipeline_imageinfo_bad_1():
    path = 'tests/datasets/shards_bad_image'
    config = ShardsDatasetConfig.from_path_and_columns(
        path,
        image_name_col="image_name",
        text_col="caption"
    )

    reader = DatasetReader()
    processor = reader.read_from_config(config)

    pipeline = FilterPipeline("test_pipeline_imageinfo_bad")
    pipeline.add_datafilter(
        ImageInfoFilter,
        {'workers': 1},
        on_error="continue"
    )
    pipeline.run(processor)

    assert len({'width', 'height', 'channels'}.intersection(set(processor.df.columns))) == 0


def test_pipeline_imageinfo_bad_2():
    path = 'tests/datasets/shards_bad_image'
    config = ShardsDatasetConfig.from_path_and_columns(
        path,
        image_name_col="image_name",
        text_col="caption"
    )

    reader = DatasetReader()
    processor = reader.read_from_config(config)

    pipeline = FilterPipeline("test_pipeline_imageinfo_bad")
    pipeline.add_datafilter(
        ImageInfoFilter,
        {'workers': 1},
        on_error="stop"
    )
    error = None
    try:
        pipeline.run(processor)
    except Exception as err:
        error = err
    assert error is not None


def test_pipeline_phash_dedup():
    path = 'tests/datasets/shards_correct'
    config = ShardsDatasetConfig.from_path_and_columns(
        path,
        image_name_col="image_name",
        text_col="caption"
    )

    reader = DatasetReader()
    processor = reader.read_from_config(config)

    pipeline = FilterPipeline("test_pipeline_phash_dedup")
    pipeline.add_datafilter(
        PHashFilter,
        {'workers': 1},
    )
    pipeline.add_deduplication(['image_phash_8'])
    pipeline.run(processor)

    assert len(processor.df) == 1


def test_pipeline_multigpu():
    path = 'tests/datasets/shards_correct'
    config = ShardsDatasetConfig.from_path_and_columns(
        path,
        image_name_col="image_name",
        text_col="caption"
    )

    reader = DatasetReader()
    processor = reader.read_from_config(config)

    pipeline = FilterPipeline("test_pipeline_phash_dedup")
    pipeline.add_datafilter(
        DummyGPUFilter,
        {'workers': 1},
        devices=["cuda:0", "cuda:1"]
    )
    pipeline.run(processor)

    assert len(processor.df) == 2 and 'dummy_label' in processor.columns
