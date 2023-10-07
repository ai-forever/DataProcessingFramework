from DPF import ShardedConfigFabric
from DPF.configs import ShardsDatasetConfig, ShardedFilesDatasetConfig


def test_shards_config():
    path = 'tests/datasets/shards_correct/'
    fabric = ShardedConfigFabric()
    config = fabric.create_t2i_config(
        path,
        format_type='shards'
    )
    print(config)
    assert isinstance(config, ShardsDatasetConfig)
    assert config.path == path
    assert len(config.datatypes) == 2


def test_files_config():
    path = 'tests/datasets/files_correct/'
    fabric = ShardedConfigFabric()
    config = fabric.create_t2i_config(
        path,
        format_type='sharded_files'
    )
    print(config)
    assert isinstance(config, ShardedFilesDatasetConfig)
    assert config.path == path
    assert len(config.datatypes) == 2

    