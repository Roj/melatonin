"Sound detection using PyRoomAcoustics (PRA)"
from dataclasses import dataclass
import logging

import scipy.signal
import numpy as np
import pyroomacoustics as pra
from rich.logging import RichHandler

from melatonin.parameters import CommonParameters


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)
log = logging.getLogger("pra")


@dataclass
class PRAParameters:
    parameters: CommonParameters
    n_sources: int
    distance: float
    dim: int
    room_dim: np.ndarray
    SNR: float
    freq_bins: np.ndarray
    

def get_microphone_signals(
    source_azimuths: np.ndarray, source_signals: np.ndarray, parameters: Parameters
):
    """Generating microphone signals by simulating an anechoic room"""
    aroom = pra.AnechoicRoom(
        parameters.dim,
        fs=parameters.sampling_frequency,
        sigma2_awgn=parameters.sigma2,
    )
    aroom.add_microphone_array(
        pra.MicrophoneArray(parameters.microphone_positions, fs=aroom.fs)
    )
    source_locations = (
        parameters.room_dim / 2
        + parameters.distance
        * np.array([np.cos(source_azimuths), np.sin(source_azimuths)]).T
    )
    for signal_i in range(parameters.n_sources):
        location, signal = source_locations[signal_i], source_signals[:, signal_i]
        aroom.add_source(location, signal=signal)

    aroom.simulate()
    return aroom.mic_array.signals


def run_algorithms(
    azimuths: list[np.ndarray],
    source_signals: np.ndarray,
    parameters: Parameters,
    algorithms: list[str],
):
    results_by_algorithm = {name: dict() for name in algorithms}
    microphone_signals = get_microphone_signals(azimuths, source_signals, parameters)

    X = np.array(
        [
            pra.transform.stft.analysis(signal, parameters.slice_size, parameters.slice_size // 2).T
            for signal in microphone_signals
        ]
    )

    for algo_name in algorithms:
        doa = pra.doa.algorithms[algo_name](
            parameters.microphone_positions,
            parameters.sampling_frequency,
            parameters.slice_size,
            c=parameters.speed_of_sound,
            num_src=len(azimuths),
        )
        doa.locate_sources(X, freq_bins=parameters.freq_bins)
        grnds = np.sort(np.rad2deg(azimuths))
        preds = np.sort(np.rad2deg(doa.azimuth_recon))
        try:
            results_by_algorithm[algo_name]["errors"] = np.linalg.norm(preds - grnds)
        except ValueError:
            logging.error(
                f"Error using algorithm {algo_name}\n"
                f"Predictions: {preds}\n"
                f"Grounds: {grnds}\n"
            )
            continue
        results_by_algorithm[algo_name]["grounds"] = grnds
        results_by_algorithm[algo_name]["predictions"] = preds
    return results_by_algorithm


def run_simulation(
    azimuths_list: list[np.ndarray], source_signals: np.ndarray, parameters: Parameters
):
    """Run simulation of repeated source detections.

    This method runs the detection algorithms (all except for FRIDA) on the source signals
    with varying direction of arrivals. The method is intended to help quantify the quality
    of the predictions.

    Parameters
    ----------
    azimuths_list : list[np.ndarray]
        Sequence of ground DoAs. Each element must be the same length and coincide with the
        parameters.n_sources.
    source_signals : list[np.ndarray]
        The signal of each source. Will be repeated across runs.
    parameters : Parameters
        Rest of parameters of the setting (such as room) and algorithms.
    """
    algorithms = sorted(set(pra.doa.algorithms.keys()) - {"FRIDA"})
    results = {algorithm: dict() for algorithm in algorithms}
    for azimuths in azimuths_list:
        partial_results = run_algorithms(
            azimuths, source_signals, parameters, algorithms
        )
        for algorithm, algorithm_results in partial_results.items():
            for metric, value in algorithm_results.items():
                if metric not in results[algorithm]:
                    results[algorithm][metric] = list()
                results[algorithm][metric].append(value)

    return results


if __name__ == "__main__":
    from rich.pretty import pprint

    room_dim = np.r_[10.0, 10.0]
    parameters = Parameters(
        n_sources=3,
        distance=3.0,
        dim=2,
        room_dim=room_dim,
        SNR=0.0,
        speed_of_sound=343.0,
        sampling_frequency=16_000,
        slice_size=256,
        freq_bins=np.arange(5, 60),
        # We use a circular array with radius 15 cm # and 12 microphones
        microphone_positions=pra.circular_2D_array(room_dim / 2, 12, 0.0, 0.15),
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
