from typing import Tuple, Optional
from enum import IntEnum


# types of resizing logic in Resizer
class ResizerModes(IntEnum):
    FIXED = 0
    MIN_SIZE = 1
    MAX_SIZE = 2


class Resizer:
    """
    Resizer class that used for different resizing strategies
    """

    def __init__(
        self,
        mode: int,
        fixed_size: Optional[Tuple[int, int]] = None,
        size: Optional[int] = None,
        downscale_only: bool = True
    ):
        self.mode = mode
        self.fixed_size = fixed_size
        self.size = size
        self.downscale_only = downscale_only

        if self.mode == ResizerModes.FIXED:
            assert len(self.fixed_size) == 2 and all(isinstance(v, int) for v in self.fixed_size)
        elif self.mode in [ResizerModes.MIN_SIZE, ResizerModes.MAX_SIZE]:
            assert isinstance(self.size, int)
        else:
            raise ValueError(f"Invalid resizer mode: {self.mode}. Use mode from ResizerModes")

    def get_new_size(self, width: int, height: int) -> Tuple[int, int]:
        if self.mode == ResizerModes.FIXED:
            new_w, new_h = self.fixed_size
        elif self.mode == ResizerModes.MIN_SIZE:
            if self.downscale_only and min(height, width) < self.size:
                new_w, new_h = width, height
            elif height >= width:
                new_w = self.size
                new_h = int(self.size * height / width)
            else:
                new_h = self.size
                new_w = int(self.size * width / height)
        else:
            if self.downscale_only and max(height, width) < self.size:
                new_w, new_h = width, height
            elif width >= height:
                new_w = self.size
                new_h = int(self.size * height / width)
            else:
                new_h = self.size
                new_w = int(self.size * width / height)

        return new_w, new_h
