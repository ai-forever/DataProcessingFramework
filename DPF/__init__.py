"""DPF framework"""

__version__ = "1.0.0"

from .configs import (
    DatasetConfig,
    FilesDatasetConfig,
    ShardedFilesDatasetConfig,
    ShardsDatasetConfig,
)
from .connectors import Connector, LocalConnector, S3Connector
from .dataset_reader import DatasetReader
from .processors import (
    DatasetProcessor,
    FilesDatasetProcessor,
    ShardedFilesDatasetProcessor,
    ShardsDatasetProcessor,
)
