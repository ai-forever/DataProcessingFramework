from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict

from DPF.validators.format_validators.errors import FileStructureError, DataFrameError, FileStructureErrorType, \
    DataFrameErrorType


@dataclass
class ValidationResult:
    filestructure_errors: List[FileStructureErrorType]
    metadata_errors: Dict[str, List[DataFrameErrorType]]

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}:"
        s += f"\nfilestructure_errors = {self.filestructure_errors}"
        s += f"\nmetadata_errors = {self.metadata_errors}"
        return s

    @property
    def total_errors(self) -> int:
        return len(self.filestructure_errors) + sum(map(len, self.metadata_errors.values()))


class Validator(ABC):

    @abstractmethod
    def validate(
        self,
        validate_filestructure: bool = True,
        validate_metadata: bool = True,
        workers: int = 1,
        pbar: bool = True
    ) -> ValidationResult:
        pass
