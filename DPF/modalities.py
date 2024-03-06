from dataclasses import dataclass
from typing import Optional


@dataclass
class DataModality:
    """Represents the modality of data

    Parameters
    ----------
    name: str
        Name of modality. Should be unique
    path_column: str
        Default column path to files with this modality
    sharded_file_name_column: str
        Default column name of filenames in shard with this modality
    column: Optional[str] = None
        Default column name. If this modality can be stored in a column use None.
    """
    name: str
    path_column: str
    sharded_file_name_column: str
    column: Optional[str] = None

    @property
    def can_be_column(self) -> bool:
        return self.column is not None

    def __hash__(self) -> int:
        return hash(self.name)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


MODALITIES = {
    'image': DataModality(
        'image', 'image_path',
        'image_name', None
    ),
    'video': DataModality(
        'video', 'video_path',
        'video_name', None
    ),
    'text': DataModality(
        'text', 'text_path',
        'text_name', 'text'
    )
}

Image = 'image'
Video = 'video'
Text = 'text'
