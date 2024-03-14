from abc import abstractmethod
from types import TracebackType
from typing import Optional, Union


class ABSWriter:
    @abstractmethod
    def save_sample(
        self,
        modality2sample_data: dict[str, tuple[str, bytes]],
        table_data: Optional[dict[str, str]] = None,
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
