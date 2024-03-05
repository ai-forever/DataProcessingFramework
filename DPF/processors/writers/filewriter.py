import traceback
from abc import abstractmethod
from typing import Dict, Optional, Tuple


class ABSWriter:
    @abstractmethod
    def save_sample(
        self,
        modality2sample_data: Dict[str, Tuple[str, bytes]],
        table_data: Optional[Dict[str, str]] = None,
    ) -> None:
        pass

    @abstractmethod
    def __enter__(self) -> "ABSWriter":
        pass

    @abstractmethod
    def __exit__(
        self,
        exception_type,
        exception_value: Optional[Exception],
        exception_traceback: traceback,
    ) -> None:
        pass
