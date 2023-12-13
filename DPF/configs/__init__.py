from .dataset_config import DatasetConfig
from .sharded_config import ShardedDatasetConfig
from .shards_config import ShardsDatasetConfig
from .sharded_files_config import ShardedFilesDatasetConfig
from .files_config import FilesDatasetConfig


def config2format(config: DatasetConfig) -> str:
    if isinstance(config, ShardsDatasetConfig):
        return "shards"
    elif isinstance(config, ShardedFilesDatasetConfig):
        return "sharded_files"
    elif isinstance(config, FilesDatasetConfig):
        return "files"
    else:
        raise ValueError(f"Unknown config type: {config}")