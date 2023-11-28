from DPF.configs import ShardsDatasetConfig, ShardedFilesDatasetConfig


def test_shards_config():
    path = 'tests/datasets/shards_correct'
    config = ShardsDatasetConfig.from_modalities(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )
    print(config)
    assert isinstance(config, ShardsDatasetConfig)
    assert config.path == path
    assert len(config.datatypes) == 2


def test_files_config():
    path = 'tests/datasets/sharded_files_correct'
    config = ShardedFilesDatasetConfig.from_modalities(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )
    print(config)
    assert isinstance(config, ShardedFilesDatasetConfig)
    assert config.path == path
    assert len(config.datatypes) == 2

    