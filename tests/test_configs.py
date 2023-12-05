from DPF.configs import ShardsDatasetConfig, ShardedFilesDatasetConfig, FilesDatasetConfig


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


def test_sharded_files_config():
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


def test_files_config():
    path = 'tests/datasets/files_correct/data.csv'
    config = FilesDatasetConfig.from_modalities(
        path,
        image_path_col="image_path",
        caption_col="caption"
    )
    print(config)
    assert isinstance(config, FilesDatasetConfig)
    assert config.table_path == path
    assert len(config.datatypes) == 2

    