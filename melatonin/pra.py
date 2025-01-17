from dataclasses import dataclass
import logging

import scipy.signal
import numpy as np
import pyroomacoustics as pra
from rich.logging import RichHandler


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("pra")

# Location of original source
azimuths = np.deg2rad([30, 120])


@dataclass
class Parameters:
    n_sources: int
    distance: float
    dim: int
    room_dim: np.ndarray
    SNR: float
    speed_of_sound: float
    sampling_frequency: int
    nfft: int
    freq_bins: np.ndarray

    @property
    def sigma2(self):
        return 10 ** (-self.SNR / 10) / (4.0 * np.pi * self.distance) ** 2


def run_simulation(
    azimuths_list: list[np.ndarray], source_signals: np.ndarray, parameters: Parameters
):
    # We use a circular array with radius 15 cm # and 12 microphones
    R = pra.circular_2D_array(parameters.room_dim / 2, 8, 0.0, 0.15)
    algo_names = sorted(set(pra.doa.algorithms.keys()) - {"FRIDA"})
    results_by_algorithm = {
        name: {
            "errors": [],
            "grounds": [],
            "predictions": [],
        }
        for name in algo_names
    }
    for azimuths in azimuths_list:
        aroom = pra.AnechoicRoom(
            parameters.dim,
            fs=parameters.sampling_frequency,
            sigma2_awgn=parameters.sigma2,
        )
        aroom.add_microphone_array(pra.MicrophoneArray(R, fs=aroom.fs))
        source_locations = (
            parameters.room_dim / 2
            + parameters.distance * np.array([np.cos(azimuths), np.sin(azimuths)]).T
        )
        for signal_i in range(parameters.n_sources):
            location, signal = source_locations[signal_i], source_signals[:, signal_i]
            aroom.add_source(location, signal=signal)

        aroom.simulate()

        X = np.array(
            [
                pra.transform.stft.analysis(
                    signal, parameters.nfft, parameters.nfft // 2
                ).T
                for signal in aroom.mic_array.signals
            ]
        )

        for algo_name in algo_names:
            doa = pra.doa.algorithms[algo_name](
                R,
                parameters.sampling_frequency,
                parameters.nfft,
                c=parameters.speed_of_sound,
                num_src=len(azimuths),
            )
            doa.locate_sources(X, freq_bins=parameters.freq_bins)
            grnds = np.sort(np.rad2deg(doa.azimuth_recon))
            preds = np.sort(np.rad2deg(azimuths))
            try:
                results_by_algorithm[algo_name]["errors"].append(
                    np.linalg.norm(preds - grnds)
                )
            except ValueError:
                logging.error(
                    f"Error using algorithm {algo_name}\n"
                    f"Predictions: {preds}\n"
                    f"Grounds: {grnds}\n"
                )
                continue
            results_by_algorithm[algo_name]["grounds"].append(grnds)
            results_by_algorithm[algo_name]["predictions"].append(preds)
    return results_by_algorithm


if __name__ == "__main__":
    from rich.pretty import pprint

    parameters = Parameters(
        n_sources=3,
        distance=3.0,
        dim=2,
        room_dim=np.r_[10.0, 10.0],
        SNR=0.0,
        speed_of_sound=343.0,
        sampling_frequency=16000,
        nfft=256,
        freq_bins=np.arange(5, 60),
    )
    t = np.linspace(0, 1, num=int(parameters.sampling_frequency * 1))
    source_signals = np.array(
        [
            # (np.sin(t*j*400*2*np.pi)).astype(np.float32)
            (scipy.signal.sawtooth(t * j * 400 * 2 * np.pi)).astype(np.float32)
            for j in range(1, parameters.n_sources + 1)
        ]
    )
    results = run_simulation([np.deg2rad([30, 90, 110])], source_signals, parameters)
    pprint(results)
