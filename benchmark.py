import logging
import typing
import numpy as np
import scipy.optimize

import numpy as np
from melatonin.detectors.base import BaseDetector
from melatonin.detectors.heraklion import HeraklionDetector, HeraklionParameters
from melatonin.detectors.pra import PRADetector, PRAParameters
from melatonin.microphones import AnechoicRoomMicrophones
from melatonin.parameters import CommonParameters
from melatonin.sources import PositionGenerator, SignalGenerator


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def align_predictions_rad(
    real_doas: np.ndarray, predicted_doas: np.ndarray
) -> np.ndarray:
    """Align predictions to grounds considering circular logic.

    Consider that a detector will not return the estimated DoAs in the same order
    that the ground DoAs are because the algorithm will only see microphone signals,
    which are a linear sum of the delayed and attenuated source signals and therefore
    has no order.


    This function will permutate the predictions in order to find the best match for
    the grounds.

    Parameters
    ----------
    real_doas : np.ndarray
        Real DoAs, in radians.
    predicted_doas : np.ndarray
        Estimated DoAs, in radians.

    Examples
    -----
    Consider for example the case where grounds is [179., 359] and the predictions are
    [0, 180]. One can see that the algorithm is pretty accurate with a MAE of 1,
    but
    1) the predictions themselves do not match the grounds
    2) we must consider the circular nature of the degrees, as the estimated angle 0
      corresponds to the true angle 359.
    The function will behave as follows:
    >>> real_doas_d = np.array([179, 359])
    >>> predicted_doas_d = np.array([0, 180])
    >>> real_doas, predicted_doas = np.deg2rad(real_doas_d), np.deg2rad(predicted_doas_d)
    >>> np.rad2deg(align_predictions(real_doas, predicted_doas))
    array([180.,   0.])
    """
    if real_doas.shape != predicted_doas.shape:
        raise ValueError(
            f"Grounds shape ({real_doas.shape}) does not match preds shape ({predicted_doas.shape})"
        )
    # We assume that they are in radians
    D = np.array(
        [
            [wrap_angle(real_i - predicted_j) for predicted_j in predicted_doas]
            for real_i in real_doas
        ]
    )
    _, col_ind = scipy.optimize.linear_sum_assignment(D)
    return predicted_doas[col_ind]


def run_multiple_algorithms(
    azimuths: np.ndarray,
    mic_fft_signals: list[np.ndarray],
    detectors: list[BaseDetector],
):
    results_by_algorithm = {detector.name: dict() for detector in detectors}

    for detector in detectors:
        preds = detector.detect(mic_fft_signals)
        grnds = np.sort(azimuths)
        preds = align_predictions_rad(grnds, preds)
        grnds, preds = np.rad2deg(grnds), np.rad2deg(preds)
        try:
            results_by_algorithm[detector.name]["errors"] = np.linalg.norm(
                preds - grnds
            )
        except ValueError:
            logging.error(
                f"Error using algorithm {detector.name}\n"
                f"Predictions: {preds}\n"
                f"Grounds: {grnds}\n"
            )
            continue
        results_by_algorithm[detector.name]["grounds"] = grnds
        results_by_algorithm[detector.name]["predictions"] = preds
    return results_by_algorithm


def run_benchmark(
    azimuths_list: list[np.ndarray],
    source_signals: np.ndarray,
    signal_generator: typing.Callable,
    detectors: list[BaseDetector],
    common_parameters: CommonParameters,
):
    # TODO: update docstring
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
    results = {detector.name: dict() for detector in detectors}
    if len(results) != len(detectors):
        raise ValueError("Repeated names for detectors!")

    for azimuths in azimuths_list:
        fft_slices = signal_generator(azimuths, source_signals, common_parameters)
        partial_results = run_multiple_algorithms(azimuths, fft_slices, detectors)
        for algorithm, algorithm_results in partial_results.items():
            for metric, value in algorithm_results.items():
                if metric not in results[algorithm]:
                    results[algorithm][metric] = list()
                results[algorithm][metric].append(value)

    return results


if __name__ == "__main__":
    # source_doas = np.deg2rad(np.array([0, 160, 200]))
    source_doas = PositionGenerator.fixed_spread_doas(3, 60)
    room_dim = np.array([5, 5])
    # Way 1: anechoic microphones
    microphone_locations = AnechoicRoomMicrophones.get_positions(
        room_dim=room_dim, number=8
    )
    print("Microphone locations: ", microphone_locations)
    awgn = AnechoicRoomMicrophones.calculate_noise_level(3, 0)  # TODO
    common_parameters_dict = dict(
        microphone_positions=microphone_locations,
        noise_level=awgn,
    )
    common_parameters = CommonParameters(**common_parameters_dict)

    pra_parameters = PRAParameters(
        **common_parameters_dict,
        n_sources=3,
        distance=None,
        dim=2,
        room_dim=room_dim,
        freq_bins=np.arange(5, 60),
        algorithm="SRP",
    )

    hk_parameters = HeraklionParameters(
        **common_parameters_dict,
        adjacent_zone=2,
        single_source_threshold=0.8,
        estimations_per_zone=4,
        histogram_bins=50,
        Q0=5,
        max_sources=3,
    )

    signals = SignalGenerator.sawtooths(
        3, common_parameters.sampling_frequency, duration=8
    )
    detectors = [
        HeraklionDetector("heraklion", hk_parameters),
        PRADetector("SRP", pra_parameters),
    ]

    def signal_generator(azimuths, source_signals, common_parameters):
        locations = PositionGenerator.doas_to_positions(
            azimuths, amplitudes=np.array([8])
        )
        mic_signals = AnechoicRoomMicrophones.get_microphone_signals(
            locations, source_signals, common_parameters
        )
        fft_slices = AnechoicRoomMicrophones.build_fft_slices(
            mic_signals, common_parameters
        )
        print("Source locations: ", locations)
        print("Signals shape: ", source_signals.shape)
        print("Mic signals shape: ", mic_signals.shape)
        print("FFT slices shape: ", fft_slices.shape)
        return fft_slices

    print(
        run_benchmark(
            [source_doas],
            signals,
            # AnechoicRoomMicrophones.get_microphone_signals,
            signal_generator,
            detectors,
            common_parameters,
        )
    )
