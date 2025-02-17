import numpy as np 
from dataclasses import dataclass

@dataclass(kw_only=True)
class CommonParameters:
    microphone_positions: np.ndarray
    slice_size: int = 2048
    sampling_frequency: int = 44100
    speed_of_sound: float = 343
    verbose: bool = False

    @property
    def overlap_size(self):
        return self.slice_size // 2

