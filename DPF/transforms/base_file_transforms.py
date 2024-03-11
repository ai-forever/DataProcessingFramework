from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, Optional

from tqdm.contrib.concurrent import process_map, thread_map

PoolOptions = Literal['processes', 'threads']


@dataclass
class TransformsFileData:
    """Represents one sample of data with file"""
    filepath: str
    metadata: dict[str, Any]


class BaseFilesTransforms(ABC):
    """Base class for all FilesTransforms"""

    def __init__(
        self,
        pool_type: PoolOptions,
        workers: int = 16,
        pbar: bool = True
    ):
        """
        Parameters
        ----------
        pool_type: PoolOptions
            Type of pool used for parallel processing. Available options: 'threads', 'processes'.
        workers: int = 16
            Number of parallel workers
        pbar: bool = True
            Whether to use progress bar
        """
        assert pool_type in ['processes', 'threads']
        self.pool_type = pool_type
        self.max_workers = workers
        self.pbar = pbar

    @property
    @abstractmethod
    def modality(self) -> str:
        """Modality name for which transformation is used"""
        pass

    @property
    @abstractmethod
    def required_metadata(self) -> list[str]:
        """List of needed column names with metadata"""
        pass

    @property
    @abstractmethod
    def metadata_to_change(self) -> list[str]:
        """List of column names that should be updated"""
        pass

    @abstractmethod
    def _process_filepath(self, data: TransformsFileData) -> TransformsFileData:
        """Method that transforms and rewrites a file

        Parameters
        ----------
        data: TransformsFileData
            Dataset sample

        Returns
        -------
        TransformsFileData
            Dataset sample with updated metadata
        """
        pass

    def run(
        self,
        paths: list[str],
        metadata_lists: Optional[dict[str, list[Any]]] = None
    ) -> list[TransformsFileData]:
        """Run transformation on files

        Parameters
        ----------
        paths: list[str]
            List of paths to files
        metadata_lists: Optional[dict[str, list[Any]]] = None
            Mapping from column name to its values

        Returns
        -------
        list[TransformsFileData]
            List of TransformsFileData with updated metadata
        """
        if self.pool_type == 'threads':
            pool_map = thread_map
        else:
            pool_map = process_map

        if len(self.required_metadata) > 0:
            assert metadata_lists is not None and isinstance(metadata_lists, dict)
            assert all(k in metadata_lists for k in self.required_metadata)
            assert all(len(paths) == len(metadata_lists[k]) for k in metadata_lists.keys())

        if metadata_lists is None:
            metadata_lists = {}

        def data_iterator() -> Iterable[TransformsFileData]:
            for i, fp in enumerate(paths):
                arg = TransformsFileData(fp, {k: v[i] for k, v in metadata_lists.items()})
                yield arg

        transformed_metadata: list[TransformsFileData] = pool_map(
            self._process_filepath,
            data_iterator(),
            total=len(paths),
            max_workers=self.max_workers,
            disable=not self.pbar
        )
        return transformed_metadata
