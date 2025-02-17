import numpy as np
import pyroomacoustics as pra
import scipy

from melatonin.parameters import CommonParameters

class MicrophoneArray:
    @staticmethod
    def get_microphone_signals(
        source_locations: np.ndarray, 
        source_signals: np.ndarray, 
        noise_level: float,
        parameters: CommonParameters
    ):
        raise NotImplementedError
    @staticmethod
    def build_fft_slices(mic_signals, parameters: CommonParameters):
        raise NotImplementedError

class CustomMicrophoneArray(MicrophoneArray):
    """A custom implementation of a microphone array"""
    @staticmethod
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

    @staticmethod
    def generate_mic_signals(
        source_locations: np.ndarray, 
        source_signals: np.ndarray, 
        parameters: CommonParameters, 
        noise_level: float, 
        verbose=True
    ):
        """0.01 noise amplitude against 0.99 signal is ~40dB SNR"""
        n_mics = len(parameters.microphone_positions)
        distances = []
        for mic_i in range(n_mics):
            distances.append(np.linalg.norm((source_locations - parameters.microphone_positions[mic_i, :]), axis=1))
        
        distances = np.array(distances)
        
        mic_signals = []
        signal_length = len(source_signals[0])
        for mic_i in range(n_mics):
            mic_signal = np.random.randn(signal_length)*noise_level
            for source_j in range(len(source_locations)):
                delay = distances[mic_i, source_j]/parameters.speed_of_sound
                start = int(parameters.sampling_frequency * delay)
                attenuation = 1/(1+np.log(delay+1)) # TODO: check
                if verbose:
                    print(f"Delay for source {source_j} to microphone {mic_i} is {delay:.4f}; attn {attenuation:.2f}")
                mic_signal[start:] += attenuation * source_signals[source_j][:signal_length - start]
            mic_signal += np.random.randn(signal_length)*noise_level
            mic_signals.append(mic_signal)
        return mic_signals
    
    @staticmethod
    def build_fft_slices(mic_signals, parameters: CommonParameters):
        mic_time_slices = []
        for mic_i, signal in enumerate(mic_signals):
            mic_time_slices.append(list())
            for start, stop in CustomMicrophoneArray.overlapping_slices(
                parameters.slice_size, parameters.overlap_size, len(signal)
            ):
                mic_time_slices[mic_i].append(signal[start:stop])

        return np.array([
            [scipy.fft.rfft(_slice) for _slice in slices]
            for slices in mic_time_slices
        ])

class AnechoicRoomMicrophones(MicrophoneArray):
    
    @staticmethod 
    def get_positions(room_dim: np.ndarray, number: int):
        # TODO: parametrize the rest?? / proxy function
        return pra.circular_2D_array(room_dim / 2, number, 0.0, 0.15)

    @staticmethod
    def calculate_noise_level(distance, SNR):
        return 10 ** (-SNR / 10) / (4.0 * np.pi * distance) ** 2
    
    @staticmethod
    def get_microphone_signals(
        source_locations: np.ndarray, 
        source_signals: np.ndarray, 
        noise_level: float,
        parameters: CommonParameters
    ):
        """Generating microphone signals by simulating an anechoic room"""
        
        aroom = pra.AnechoicRoom(
            2,
            fs=parameters.sampling_frequency,
            sigma2_awgn=noise_level,
        )
        aroom.add_microphone_array(
            pra.MicrophoneArray(parameters.microphone_positions, fs=aroom.fs)
        )
        
        for signal_i in range(len(source_locations)):
            location, signal = source_locations[signal_i], source_signals[:, signal_i]
            aroom.add_source(location, signal=signal)

        aroom.simulate()
        return aroom.mic_array.signals

    @staticmethod
    def build_fft_slices(mic_signals: np.ndarray, parameters: CommonParameters):
        """FFT slices using PRA's Short Time FT"""
        return np.array(
            [
                pra.transform.stft.analysis(signal, parameters.slice_size, parameters.overlap_size).T
                for signal in mic_signals
            ]
        )