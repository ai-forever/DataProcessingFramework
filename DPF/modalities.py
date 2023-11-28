from typing import Optional
from dataclasses import dataclass


@dataclass
class Modality:
    """Describes the modality of data"""
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