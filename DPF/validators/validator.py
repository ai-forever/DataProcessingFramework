from abc import ABC, abstractmethod
from dataclasses import dataclass

from DPF.validators.errors import DataFrameErrorType, FileStructureErrorType


@dataclass
class ValidationResult:
    """Result of dataset validation

    Parameters
    ----------
    filestructure_errors: list[FileStructureErrorType]
        Dataset filestructure errors
    metadata_errors: dict[str, list[DataFrameErrorType]]
        Errors in metadata dataframes
    """
    filestructure_errors: list[FileStructureErrorType]
    metadata_errors: dict[str, list[DataFrameErrorType]]

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
        """Renames columns in files of a dataset

        Parameters
        ----------
        validate_filestructure: bool = True
            Whether to validate the filestructure of a dataset
        validate_metadata: bool = True
            Whether to validate the metadata (dataframes) of a dataset
        workers: int = 1
            Number of parallel threads for validation
        pbar: bool = True
            Whether to show progress bar or not

        Returns
        -------
        ValidationResult
            Information about errors and validation result
        """
        pass
