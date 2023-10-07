from abc import ABC, abstractmethod


class ValidationResult:
    pass


class Validator(ABC):

    @abstractmethod
    def validate(
        self,
        validate_filestructure: bool = True,
        validate_dataframes: bool = True,
        threads: int = 16,
        pbar: bool = True
    ) -> ValidationResult:
        pass
