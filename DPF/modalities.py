from dataclasses import dataclass
from typing import Optional


@dataclass
class Modality:
    """Represents the modality of data

    Parameters
    ----------
    key: str
        Name of modality. Should be unique
    can_be_file: bool
        Can this modality be in file
    can_be_column: bool
        Can this modality be in a column of "csv" file
    column: Optional[str] = None
        Default column name if can_be_column=True
    sharded_file_name_column: Optional[str] = None
        Default column name of filenames with this modality. Only if can_be_file=True
    path_column: Optional[str] = None
        Default column path to files with this modality. Only if can_be_file=True
    """
    key: str
    can_be_file: bool
    can_be_column: bool
    column: Optional[str] = None
    sharded_file_name_column: Optional[str] = None
    path_column: Optional[str] = None

    def __hash__(self) -> int:
        return hash(self.key)

    def __str__(self) -> str:
        return self.key

    def __repr__(self) -> str:
        return self.key


MODALITIES = {
    'image': Modality(
        'image', True, False, None,
        'image_name', 'image_path'
    ),
    'video': Modality(
        'video', True, False, None,
        'video_name', 'video_path'
    ),
    'text': Modality(
        'text', True, True, 'text',
        'text_name', 'text_path'
    )
}

Image = 'image'
Video = 'video'
Text = 'text'
