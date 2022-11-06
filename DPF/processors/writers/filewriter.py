import abc
import logging
import traceback
from typing import Optional, Dict

class FileWriter:
    @abc.abstractmethod
    def save_file(
        self, 
        file_bytes: bytes,
        image_ext: Optional[str] = None,
        file_data: Optional[Dict[str, str]] = None
    ) -> None:
        pass

    @abc.abstractmethod
    def __enter__(self) -> "FileWriter":
        pass

    @abc.abstractmethod
    def __exit__(
        self, exception_type, exception_value: Optional[Exception], exception_traceback: traceback
    ) -> None:
        pass