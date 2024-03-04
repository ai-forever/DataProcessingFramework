import shutil
import os
from DPF import DatasetReader
from DPF.configs import ShardsDatasetConfig, ShardedFilesDatasetConfig, FilesDatasetConfig


def test_shards_to_shards():
    path = 'tests/datasets/shards_correct'
    config = ShardsDatasetConfig.from_modalities(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    processor = reader.from_config(config)
    new_dir = 'test_shards/'
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    processor.to_shards(
        new_dir,
        keys_mapping={'text': 'caption'},
        workers=1
    )

    config = ShardsDatasetConfig.from_modalities(
        new_dir.rstrip('/'),
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    processor = reader.from_config(config)
    result = processor.validate()
    assert result.total_errors == 0

    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)


def test_shards_to_sharded_files():
    path = 'tests/datasets/shards_correct'
    config = ShardsDatasetConfig.from_modalities(
        path,
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    processor = reader.from_config(config)
    new_dir = 'test_sharded_files/'
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    processor.to_sharded_files(
        new_dir,
        keys_mapping={'text': 'caption'},
        workers=1
    )

    config = ShardedFilesDatasetConfig.from_modalities(
        new_dir.rstrip('/'),
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    processor = reader.from_config(config)
    result = processor.validate()
    assert result.total_errors == 0

    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)


def test_files_to_shards():
    path = 'tests/datasets/files_correct/data.csv'
    config = FilesDatasetConfig.from_modalities(
        path,
        image_path_col="image_path",
        caption_col="caption"
    )

    reader = DatasetReader()
    processor = reader.from_config(config)
    new_dir = 'test_shards/'
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    processor.to_shards(
        new_dir.rstrip('/'),
        keys_mapping={'text': 'caption'},
        workers=1
    )

    config = ShardsDatasetConfig.from_modalities(
        new_dir.rstrip('/'),
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    processor = reader.from_config(config)
    result = processor.validate()
    assert result.total_errors == 0

    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)


def test_files_to_sharded_files():
    path = 'tests/datasets/files_correct/data.csv'
    config = FilesDatasetConfig.from_modalities(
        path,
        image_path_col="image_path",
        caption_col="caption"
    )

    reader = DatasetReader()
    processor = reader.from_config(config)
    new_dir = 'test_sharded_files/'
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    processor.to_sharded_files(
        new_dir.rstrip('/'),
        keys_mapping={'text': 'caption'},
        workers=1
    )

    config = ShardedFilesDatasetConfig.from_modalities(
        new_dir.rstrip('/'),
        image_name_col="image_name",
        caption_col="caption"
    )

    reader = DatasetReader()
    processor = reader.from_config(config)
    result = processor.validate()
    assert result.total_errors == 0

    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)