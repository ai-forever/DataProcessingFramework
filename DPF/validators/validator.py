from abc import ABC, abstractmethod


class ValidationResult:
    pass


class Validator(ABC):

    @abstractmethod
    def validate(
        self,
        validate_filestructure: bool = True,
        validate_dataframes: bool = True,
        workers: int = 4,
        pbar: bool = True
    ) -> ValidationResult:
        pass
