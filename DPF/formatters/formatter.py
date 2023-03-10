from abc import ABC
from DPF.filesystems import LocalFileSystem, S3FileSystem


class Formatter(ABC):
    """
    Abstract class for all formatters
    """

    def __init__(
            self,
            filesystem: str = 'local',
            **filesystem_kwargs
    ):
        """
        Parameters
        ----------
        filesystem: str
            'local' | 's3', type of filesystem to use
        **filesystem_kwargs
            kwargs for corresponding DPF.filesystems.FileSystem class
        """
        if filesystem == 'local':
            self.filesystem = LocalFileSystem()
        elif filesystem == 's3':
            self.filesystem = S3FileSystem(**filesystem_kwargs)
        else:
            raise NotImplementedError(f"Unknown filesystem format: {filesystem}")
