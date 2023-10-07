from DPF import ShardedConfigFabric, DatasetReader
from DPF.configs import ShardsDatasetConfig
from DPF.processors import ShardsDatasetProcessor, ShardedFilesDatasetProcessor


def test_shards_reader():
    path = 'tests/datasets/shards_correct/'
    fabric = ShardedConfigFabric()
    config = fabric.create_t2i_config(
        path,
        format_type='shards'
    )

    reader = DatasetReader()
    dataset = reader.from_config(config)
    assert isinstance(dataset, ShardsDatasetProcessor)
    assert len(dataset.df) == 2


def test_shards_wrong_columns():
    path = 'tests/datasets/shards_wrong_columns/'
    fabric = ShardedConfigFabric()
    config = fabric.create_t2i_config(
        path,
        format_type='shards'
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
    fabric = ShardedConfigFabric()
    config = fabric.create_t2i_config(
        path,
        format_type='shards'
    )
    reader = DatasetReader()
    dataset = reader.from_config(config)
    assert dataset.df.shape == (2, 3)


def test_files_reader():
    path = 'tests/datasets/files_correct/'
    fabric = ShardedConfigFabric()
    config = fabric.create_t2i_config(
        path,
        format_type='sharded_files'
    )

    reader = DatasetReader()
    dataset = reader.from_config(config)
    assert isinstance(dataset, ShardedFilesDatasetProcessor)
    assert len(dataset.df) == 2


def test_files_wrong_columns():
    path = 'tests/datasets/files_wrong_columns/'
    fabric = ShardedConfigFabric()
    config = fabric.create_t2i_config(
        path,
        format_type='sharded_files'
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
    path = 'tests/datasets/files_wrong_folder/'
    fabric = ShardedConfigFabric()
    config = fabric.create_t2i_config(
        path,
        format_type='sharded_files'
    )
    reader = DatasetReader()
    dataset = reader.from_config(config)
    assert dataset.df.shape == (2, 3)
