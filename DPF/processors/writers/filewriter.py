from types import TracebackType
from abc import abstractmethod
from typing import Dict, Optional, Tuple, Union


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
        exception_type: Union[type[BaseException], None],
        exception_value: Union[BaseException, None],
        exception_traceback: Union[TracebackType, None],
    ) -> None:
        pass
