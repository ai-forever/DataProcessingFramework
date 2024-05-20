from .dataset_config import DatasetConfig
from .files_config import FilesDatasetConfig
from .sharded_config import ShardedDatasetConfig
from .sharded_files_config import ShardedFilesDatasetConfig
from .shards_config import ShardsDatasetConfig


def config2format(config: DatasetConfig) -> str:
    if isinstance(config, ShardsDatasetConfig):
        return "shards"
    elif isinstance(config, ShardedFilesDatasetConfig):
        return "sharded_files"
    elif isinstance(config, FilesDatasetConfig):
        return "files"
    else:
        raise ValueError(f"Unknown config type: {config}")
