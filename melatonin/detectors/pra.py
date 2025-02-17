"Sound detection using PyRoomAcoustics (PRA)"
from dataclasses import dataclass
import logging
import typing

import scipy.signal
import numpy as np
from melatonin.microphones import AnechoicRoomMicrophones
import pyroomacoustics as pra
from rich.logging import RichHandler

from melatonin.parameters import CommonParameters


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("pra")


@dataclass(kw_only=True)
class PRAParameters(CommonParameters):
    n_sources: int
    distance: float
    dim: int
    room_dim: np.ndarray
    SNR: float
    freq_bins: np.ndarray
    algorithm: typing.Literal["SPR", "MUSIC", "NormMUSIC", "CSSM"]

class PRADetector:
    def __init__(self, name: str, parameters: PRAParameters):
        self.name = name
        self.parameters = parameters 
        self.doa = None

    def detect(self, microphones_fft_slices):
        self.doa = pra.doa.algorithms[self.parameters.algorithm](
            self.parameters.microphone_positions,
            self.parameters.sampling_frequency,
            self.parameters.slice_size,
            c=self.parameters.speed_of_sound,
            num_src=self.parameters.n_sources,
        )
        self.doa.locate_sources(microphones_fft_slices, freq_bins=self.parameters.freq_bins)
        
        return self.doa.azimuth_recon



if __name__ == "__main__":
    from rich.pretty import pprint

    room_dim = np.r_[10.0, 10.0]
    parameters = PRAParameters(
        # TODO add this to the source positions
        microphone_positions=AnechoicRoomMicrophones.get_positions(room_dim, 12),
        slice_size=256,
        sampling_frequency=16_000,
        speed_of_sound=343.0,
        verbose=False,
        n_sources=3,
        distance=3.0,
        dim=2,
        room_dim=room_dim,
        SNR=0.0,        
        freq_bins=np.arange(5, 60),
        
        
    )
    t = np.linspace(0, 1, num=int(parameters.sampling_frequency * 1))
    source_signals = np.array(
        [
            (scipy.signal.sawtooth(t * j * 400 * 2 * np.pi)).astype(np.float32)
            for j in range(1, parameters.n_sources + 1)
        ]
    )
    results = run_simulation([np.deg2rad([30, 90, 110])], source_signals, parameters)
    pprint(results)
