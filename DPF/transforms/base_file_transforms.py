from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Iterable
from tqdm.contrib.concurrent import process_map, thread_map
from dataclasses import dataclass


@dataclass
class TransformsFileArguments:
    filepath: str
    metadata: Dict[str, Any]


class BaseFilesTransforms(ABC):

    def __init__(
        self,
        pool_type: str = 'processes',
        max_workers: int = 16,
        pbar: bool = True
    ):
        assert pool_type in ['processes', 'threads']
        self.pool_type = pool_type
        self.max_workers = max_workers
        self.pbar = pbar

    @property
    @abstractmethod
    def modality(self) -> str:
        pass

    @property
    @abstractmethod
    def required_metadata(self) -> List[str]:
        pass

    @abstractmethod
    def _process_filepath(self, data: TransformsFileArguments):
        pass

    def run(self, paths: List[str], metadata_lists: Optional[Dict[str, List[Any]]] = None):
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

        def data_iterator() -> Iterable[TransformsFileArguments]:
            for i, fp in enumerate(paths):
                arg = TransformsFileArguments(fp, {k: v[i] for k, v in metadata_lists.items()})
                yield arg

        pool_map(
            self._process_filepath,
            data_iterator(),
            total=len(paths),
            max_workers=self.max_workers,
            disable=not self.pbar
        )