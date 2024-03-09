from enum import IntEnum
from typing import Tuple


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
        mode: ResizerModes,
        fixed_size: Tuple[int, int] = (512, 512),
        size: int = 512,
        downscale_only: bool = True
    ):
        self.mode = mode
        assert self.mode in [ResizerModes.FIXED, ResizerModes.MIN_SIZE, ResizerModes.MAX_SIZE], \
            f"Invalid resizer mode: {self.mode}. Use mode from ResizerModes"

        self.fixed_size = fixed_size
        self.size = size
        self.downscale_only = downscale_only

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
