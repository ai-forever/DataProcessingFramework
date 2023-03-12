from abc import abstractmethod
import traceback
from typing import Optional, Dict


class FileWriter:
    @abstractmethod
    def save_file(
        self,
        file_bytes: bytes,
        image_ext: Optional[str] = None,
        file_data: Optional[Dict[str, str]] = None,
    ) -> None:
        pass

    @abstractmethod
    def __enter__(self) -> "FileWriter":
        pass

    @abstractmethod
    def __exit__(
        self,
        exception_type,
        exception_value: Optional[Exception],
        exception_traceback: traceback,
    ) -> None:
        pass
