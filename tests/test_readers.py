import os

from DPF import DatasetReader
from DPF.configs import (
    FilesDatasetConfig,
    ShardedFilesDatasetConfig,
    ShardsDatasetConfig,
)
from DPF.processors import (
    FilesDatasetProcessor,
    ShardedFilesDatasetProcessor,
    ShardsDatasetProcessor,
)


def test_shards_reader():
    path = 'tests/datasets/shards_correct'
    config = ShardsDatasetConfig.from_path_and_columns(
        path,
        image_name_col="image_name",
        text_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.read_from_config(config)
    assert isinstance(dataset, ShardsDatasetProcessor)
    assert len(dataset.df) == 2


def test_shards_wrong_columns():
    path = 'tests/datasets/shards_wrong_columns'
    config = ShardsDatasetConfig.from_path_and_columns(
        path,
        image_name_col="image_name",
        text_col="caption"
    )

    reader = DatasetReader()
    dataset = None
    try:
        dataset = reader.read_from_config(config)
    except Exception as err:
        assert isinstance(err, AssertionError)

    assert dataset is None

    dataset = reader.read_from_config(config, validate_columns=False)
    assert dataset.df.shape == (4, 4)


def test_shards_wrong_tar():
    path = 'tests/datasets/shards_wrong_tar'
    config = ShardsDatasetConfig.from_path_and_columns(
        path,
        image_name_col="image_name",
        text_col="caption"
    )
    reader = DatasetReader()
    dataset = reader.read_from_config(config)
    assert dataset.df.shape == (2, 3)


def test_sharded_files_reader():
    path = 'tests/datasets/sharded_files_correct'
    config = ShardedFilesDatasetConfig.from_path_and_columns(
        path,
        image_name_col="image_name",
        text_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.read_from_config(config)
    assert isinstance(dataset, ShardedFilesDatasetProcessor)
    assert len(dataset.df) == 2


def test_sharded_files_wrong_columns():
    path = 'tests/datasets/sharded_files_wrong_columns'
    config = ShardedFilesDatasetConfig.from_path_and_columns(
        path,
        image_name_col="image_name",
        text_col="caption"
    )

    reader = DatasetReader()
    dataset = None
    try:
        dataset = reader.read_from_config(config)
    except Exception as err:
        assert isinstance(err, AssertionError)

    assert dataset is None

    dataset = reader.read_from_config(config, validate_columns=False)
    assert dataset.df.shape == (4, 4)


def test_sharded_files_wrong_tar():
    path = 'tests/datasets/sharded_files_wrong_folder'
    config = ShardedFilesDatasetConfig.from_path_and_columns(
        path,
        image_name_col="image_name",
        text_col="caption"
    )
    reader = DatasetReader()
    dataset = reader.read_from_config(config)
    assert dataset.df.shape == (2, 3)


def test_files_reader():
    path = 'tests/datasets/files_correct/data.csv'
    config = FilesDatasetConfig.from_path_and_columns(
        path,
        image_path_col="image_path",
        text_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.read_from_config(config)
    assert isinstance(dataset, FilesDatasetProcessor)
    assert len(dataset.df) == 2

    assert all(dataset.df['image_path'].apply(lambda x: os.path.exists(x)).tolist())
