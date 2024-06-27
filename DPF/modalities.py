from dataclasses import dataclass
from typing import Literal, Optional

ModalityName = Literal["image", "video", "text", "audio"]


@dataclass
class DataModality:
    """Represents the modality of data

    Parameters
    ----------
    name: ModalityName
        Name of modality. Should be unique
    path_column: str
        Default column path to files with this modality
    sharded_file_name_column: str
        Default column name of filenames in shard with this modality
    column: Optional[str] = None
        Default column name. If this modality can be stored in a column use None.
    """
    name: ModalityName
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
    'audio': DataModality(
        'audio', 'audio_path',
        'audio_name', None
    ),
    'text': DataModality(
        'text', 'text_path',
        'text_name', 'text'
    )
}
