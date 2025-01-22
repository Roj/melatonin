import numpy as np
import pyroomacoustics as pra
import scipy

from melatonin.parameters import Parameters

def overlapping_slices(
    slice_size: int, overlap_size: int, max_value: int
) -> tuple[int, int]:
    """Generate windows of slice_size guaranteeing overlap.

    Generates the following slices:
    0: [0, slice_size]
    1: [slice_size - overlap_size, 2*slice_size - overlap_size]
    2: [2*slice_size - 2*overlap_size, 3*slice_size - 2*overlap_size]
    This way each chunk is of slice_size length, but there is overlap_size
    shared between successive chunks

    Parameters
    ----------
    slice_size : int
        Size of each chunk.
    overlap_size : int
        Overlap between a chunk and the previous one.
    max_value : int
        Maximum value of the chunk stop.

    Yields
    -------
    tuple[int, int]
        Start and stop of current chunk.
    """

    start, stop = 0, slice_size

    while stop < max_value:
        yield start, stop
        start += slice_size - overlap_size
        stop = start + slice_size

def generate_mic_signals(source_positions, different_exhausts, parameters: Parameters, noise=True, verbose=True):
    n_mics = len(parameters.microphone_positions)
    distances = []
    for mic_i in range(n_mics):
        distances.append(np.linalg.norm((source_positions - parameters.microphone_positions[mic_i, :]), axis=1))
    
    distances = np.array(distances)
    
    time_window = 8
    mic_signals = []

    for mic_i in range(n_mics):
        mic_signal = None
        for source_j in range(len(source_positions)):
            delay = distances[mic_i, source_j]/parameters.speed_of_sound
            start = int(parameters.sampling_frequency * delay)
            attenuation = 1/(1+np.log(delay+1)) # TODO: check
            delayed_signal = np.zeros(int(time_window * parameters.sampling_frequency))
            delayed_signal[start:] = attenuation * different_exhausts[source_j][:int(time_window*parameters.sampling_frequency-start)]
            if verbose:
                print(f"Delay for source {source_j} to microphone {mic_i} is {delay:.4f}; attn {attenuation:.2f}")
            if mic_signal is None:
                mic_signal = delayed_signal
            else:
                mic_signal += delayed_signal
        if noise:
            # 0.01 noise amplitude against 0.99 signal is ~40dB SNR
            mic_signal += np.random.randn(int(time_window * parameters.sampling_frequency))*0.01
        mic_signals.append(mic_signal[int(parameters.sampling_frequency*1.1):])
    return mic_signals

def build_fft_slices(mic_signals, parameters: Parameters):
    mic_time_slices = []
    for mic_i, signal in enumerate(mic_signals):
        mic_time_slices.append(list())
        for start, stop in overlapping_slices(
            parameters.slice_size, parameters.overlap_size, len(signal)
        ):
            mic_time_slices[mic_i].append(signal[start:stop])

    return [
        [scipy.fft.rfft(_slice) for _slice in slices]
        for slices in mic_time_slices
    ]


def get_microphone_signals(
    source_azimuths: np.ndarray, 
    source_signals: np.ndarray, 
    distance: float,
    SNR: float,
    room_dim: float,
    parameters: Parameters
):
    """Generating microphone signals by simulating an anechoic room"""
    sigma2 = 10 ** (-SNR / 10) / (4.0 * np.pi * distance) ** 2
    aroom = pra.AnechoicRoom(
        2,
        fs=parameters.sampling_frequency,
        sigma2_awgn=sigma2,
    )
    aroom.add_microphone_array(
        pra.MicrophoneArray(parameters.microphone_positions, fs=aroom.fs)
    )
    source_locations = (
        room_dim / 2
        + distance
        * np.array([np.cos(source_azimuths), np.sin(source_azimuths)]).T
    )
    for signal_i in range(len(source_azimuths)):
        location, signal = source_locations[signal_i], source_signals[:, signal_i]
        aroom.add_source(location, signal=signal)

    aroom.simulate()
    return aroom.mic_array.signals