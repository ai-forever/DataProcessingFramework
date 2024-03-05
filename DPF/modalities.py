from dataclasses import dataclass
from typing import Optional


@dataclass
class Modality:
    """Represents the modality of data

    Parameters
    ----------
    key: str
        Name of modality. Should be unique
    path_column: str
        Default column path to files with this modality
    sharded_file_name_column: str
        Default column name of filenames in shard with this modality
    can_be_column: bool
        Can this modality be in a column of "csv" file
    column: Optional[str] = None
        Default column name if can_be_column=True
    """
    key: str
    path_column: str
    sharded_file_name_column: str
    can_be_column: bool
    column: Optional[str] = None

    def __hash__(self) -> int:
        return hash(self.key)

    def __str__(self) -> str:
        return self.key

    def __repr__(self) -> str:
        return self.key


MODALITIES = {
    'image': Modality(
        'image', 'image_path',
        'image_name', False, None
    ),
    'video': Modality(
        'video', 'video_path',
        'video_name', False, None
    ),
    'text': Modality(
        'text', 'text_path',
        'text_name', True, 'text'
    )
}

Image = 'image'
Video = 'video'
Text = 'text'
