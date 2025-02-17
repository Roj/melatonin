
import numpy as np
from melatonin.parameters import CommonParameters


class BaseDetector:
    def __init__(self, name: str, parameters: CommonParameters):
        self.name = name
        self.parameters = parameters
    
    def detect(self, mic_fft_slices: np.ndarray):
        raise NotImplementedError