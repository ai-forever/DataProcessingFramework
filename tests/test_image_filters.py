from DPF import DatasetReader
from DPF.configs import (
    FilesDatasetConfig,
    ShardedFilesDatasetConfig,
    ShardsDatasetConfig,
)
from DPF.filters.images.hash_filters import PHashFilter
from DPF.filters.images.info_filter import ImageInfoFilter


def test_shards_info_filter():
    path = 'tests/datasets/shards_correct'
    config = ShardsDatasetConfig.from_modalities(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.from_config(config)
    filter_ = ImageInfoFilter(workers=1)
    dataset.apply_data_filter(filter_)

    assert dataset.df['is_correct'].all()
    assert dataset.df['error'].isna().all()
    assert not dataset.df['width'].isna().any()
    assert not dataset.df['height'].isna().any()
    assert not dataset.df['channels'].isna().any()


def test_shards_phash_filter():
    path = 'tests/datasets/shards_correct'
    config = ShardsDatasetConfig.from_modalities(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.from_config(config)
    filter_ = PHashFilter(workers=1)
    dataset.apply_data_filter(filter_)

    assert not dataset.df['image_phash_8'].isna().any()


def test_shards_bad_image_info_filter():
    path = 'tests/datasets/shards_bad_image'
    config = ShardsDatasetConfig.from_modalities(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )
    reader = DatasetReader()
    dataset = reader.from_config(config)
    filter_ = ImageInfoFilter(workers=1)

    error = None
    try:
        dataset.apply_data_filter(filter_)
    except KeyError as err:
        error = err
    except FileNotFoundError as err:
        error = err
    assert error is not None and isinstance(error, (KeyError, FileNotFoundError))

    dataset.apply_data_filter(filter_, validate_filter_result=False, return_none_on_error=True)
    dataset.df['is_correct'] = dataset.df['is_correct'].fillna(False)
    assert len(dataset.df[dataset.df['is_correct']]) == 2 and len(dataset.df) == 4


def test_sharded_files_info_filter():
    path = 'tests/datasets/sharded_files_correct'
    config = ShardedFilesDatasetConfig.from_modalities(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.from_config(config)
    filter_ = ImageInfoFilter(workers=1)
    dataset.apply_data_filter(filter_)

    assert dataset.df['is_correct'].all()
    assert dataset.df['error'].isna().all()
    assert not dataset.df['width'].isna().any()
    assert not dataset.df['height'].isna().any()
    assert not dataset.df['channels'].isna().any()


def test_sharded_files_phash_filter():
    path = 'tests/datasets/sharded_files_correct'
    config = ShardedFilesDatasetConfig.from_modalities(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.from_config(config)
    filter_ = PHashFilter(workers=1)
    dataset.apply_data_filter(filter_)

    assert not dataset.df['image_phash_8'].isna().any()


def test_files_info_filter():
    path = 'tests/datasets/files_correct/data.csv'
    config = FilesDatasetConfig.from_modalities(
        path,
        image_path_col="image_path",
        caption_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.from_config(config)
    filter_ = ImageInfoFilter(workers=1)
    dataset.apply_data_filter(filter_)

    assert dataset.df['is_correct'].all()
    assert dataset.df['error'].isna().all()
    assert not dataset.df['width'].isna().any()
    assert not dataset.df['height'].isna().any()
    assert not dataset.df['channels'].isna().any()


def test_files_phash_filter():
    path = 'tests/datasets/files_correct/data.csv'
    config = FilesDatasetConfig.from_modalities(
        path,
        image_path_col="image_path",
        caption_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.from_config(config)
    filter_ = PHashFilter(workers=1)
    dataset.apply_data_filter(filter_)

    assert not dataset.df['image_phash_8'].isna().any()


def test_files_bad_image():
    path = 'tests/datasets/files_bad_image/data.csv'
    config = FilesDatasetConfig.from_modalities(
        path,
        image_path_col="image_path",
        caption_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.from_config(config)
    filter_ = ImageInfoFilter(workers=1)

    error = None
    try:
        dataset.apply_data_filter(filter_)
    except KeyError as err:
        error = err
    except FileNotFoundError as err:
        error = err
    assert error is not None and isinstance(error, (KeyError, FileNotFoundError))

    dataset.apply_data_filter(filter_, validate_filter_result=False, return_none_on_error=True)
    dataset.df['is_correct'] = dataset.df['is_correct'].fillna(False)
    assert len(dataset.df[dataset.df['is_correct']]) == 2 and len(dataset.df) == 4
