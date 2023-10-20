from DPF import ShardedConfigFabric, DatasetReader
from DPF.validators.format_validators import (
    ShardsValidator, ShardedFilesValidator, IsNotKeyError, MissedColumnsError, MissingValueError
)


def test_shards_reader():
    path = 'tests/datasets/shards_correct/'
    fabric = ShardedConfigFabric()
    config = fabric.create_t2i_config(
        path,
        format_type='shards'
    )

    reader = DatasetReader()
    processor = reader.from_config(config)

    result = processor.validate()
    assert result.total_errors == 0


def test_shards_wrong_columns():
    path = 'tests/datasets/shards_wrong_columns/'
    fabric = ShardedConfigFabric()
    config = fabric.create_t2i_config(
        path,
        format_type='shards'
    )

    reader = DatasetReader()
    processor = reader.from_config(config, validate_columns=False)

    result = processor.validate()
    assert len(result.dataframe_errors) == 0
    assert len(result.filestructure_errors) == 1
    assert isinstance(result.filestructure_errors[0], IsNotKeyError)

    result = processor.validate(columns_to_check=['image_name', 'caption', 'test'])
    assert len(result.dataframe_errors) == 1
    assert isinstance(result.dataframe_errors[path+'0.csv'][0], MissedColumnsError)
    assert len(result.filestructure_errors) == 1
    assert isinstance(result.filestructure_errors[0], IsNotKeyError)


def test_shards_wrong_tar():
    path = 'tests/datasets/shards_wrong_tar/'
    fabric = ShardedConfigFabric()
    config = fabric.create_t2i_config(
        path,
        format_type='shards'
    )
    reader = DatasetReader()
    processor = reader.from_config(config)

    result = processor.validate()

    print(result)
    assert len(result.dataframe_errors) == 1
    assert isinstance(result.dataframe_errors[path + '0.csv'][0], MissingValueError)
    assert len(result.filestructure_errors) == 0


def test_files_reader():
    path = 'tests/datasets/sharded_files_correct/'
    fabric = ShardedConfigFabric()
    config = fabric.create_t2i_config(
        path,
        format_type='sharded_files'
    )

    reader = DatasetReader()
    processor = reader.from_config(config)

    result = processor.validate()
    assert result.total_errors == 0


def test_files_wrong_columns():
    path = 'tests/datasets/sharded_files_wrong_columns/'
    fabric = ShardedConfigFabric()
    config = fabric.create_t2i_config(
        path,
        format_type='sharded_files'
    )

    reader = DatasetReader()
    processor = reader.from_config(config, validate_columns=False)

    result = processor.validate()
    assert len(result.dataframe_errors) == 0
    assert len(result.filestructure_errors) == 1
    assert isinstance(result.filestructure_errors[0], IsNotKeyError)

    result = processor.validate(columns_to_check=['image_name', 'caption', 'test'])
    assert len(result.dataframe_errors) == 1
    assert isinstance(result.dataframe_errors[path+'0.csv'][0], MissedColumnsError)
    assert len(result.filestructure_errors) == 1
    assert isinstance(result.filestructure_errors[0], IsNotKeyError)


def test_files_wrong_tar():
    path = 'tests/datasets/sharded_files_wrong_folder/'
    fabric = ShardedConfigFabric()
    config = fabric.create_t2i_config(
        path,
        format_type='sharded_files'
    )
    reader = DatasetReader()
    processor = reader.from_config(config)

    result = processor.validate()

    print(result)
    assert len(result.dataframe_errors) == 1
    assert isinstance(result.dataframe_errors[path + '0.csv'][0], MissingValueError)
    assert len(result.filestructure_errors) == 0
