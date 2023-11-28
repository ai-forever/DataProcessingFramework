from DPF import DatasetReader
from DPF.configs import ShardsDatasetConfig, ShardedFilesDatasetConfig
from DPF.processors import ShardsDatasetProcessor, ShardedFilesDatasetProcessor


def test_shards_reader():
    path = 'tests/datasets/shards_correct/'
    config = ShardsDatasetConfig.from_modalities(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.from_config(config)
    assert isinstance(dataset, ShardsDatasetProcessor)
    assert len(dataset.df) == 2


def test_shards_wrong_columns():
    path = 'tests/datasets/shards_wrong_columns/'
    config = ShardsDatasetConfig.from_modalities(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    dataset = None
    try:
        dataset = reader.from_config(config)
    except Exception as err:
        assert isinstance(err, AssertionError)

    assert dataset is None

    dataset = reader.from_config(config, validate_columns=False)
    assert dataset.df.shape == (4, 4)


def test_shards_wrong_tar():
    path = 'tests/datasets/shards_wrong_tar/'
    config = ShardsDatasetConfig.from_modalities(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )
    reader = DatasetReader()
    dataset = reader.from_config(config)
    assert dataset.df.shape == (2, 3)


def test_files_reader():
    path = 'tests/datasets/sharded_files_correct/'
    config = ShardedFilesDatasetConfig.from_modalities(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    dataset = reader.from_config(config)
    assert isinstance(dataset, ShardedFilesDatasetProcessor)
    assert len(dataset.df) == 2


def test_files_wrong_columns():
    path = 'tests/datasets/sharded_files_wrong_columns/'
    config = ShardedFilesDatasetConfig.from_modalities(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    dataset = None
    try:
        dataset = reader.from_config(config)
    except Exception as err:
        assert isinstance(err, AssertionError)

    assert dataset is None

    dataset = reader.from_config(config, validate_columns=False)
    assert dataset.df.shape == (4, 4)


def test_files_wrong_tar():
    path = 'tests/datasets/sharded_files_wrong_folder/'
    config = ShardedFilesDatasetConfig.from_modalities(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )
    reader = DatasetReader()
    dataset = reader.from_config(config)
    assert dataset.df.shape == (2, 3)
