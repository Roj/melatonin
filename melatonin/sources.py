import time
import numpy as np
import pyaudio
import scipy.signal
import logging
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

log = logging.getLogger("detector")



class PositionGenerator: 
    """Helper class to create source configurations"""
    @staticmethod
    def sources_uniform_doas(num):
        return np.random.random(num)*2*np.pi
    
    @staticmethod
    def sources_uniform(num, amplitude=6):
        doas = PositionGenerator.sources_uniform_doas(num)
        return np.stack([np.cos(doas), np.sin(doas)], axis=1)*amplitude

    @staticmethod 
    def sources_on_circle(num, distance, offset):
        doas = PositionGenerator.sources_uniform_doas(num)
        return np.stack([np.cos(doas), np.sin(doas)], axis=1)*distance + offset
    @staticmethod
    def fixed_spread_doas(num, spread_deg):
        # First offset can be at most at the spread angle
        spread_rad = np.deg2rad(spread_deg)
        if spread_rad * num > 2*np.pi:
            raise ValueError(f"{num} * {spread_deg} > 360!!")
        offset = np.random.random()*(2*np.pi - spread_rad*num)
        return np.array([offset + i*spread_rad for i in range(num)])
    
    @staticmethod
    def fixed_spread(num: int, spread_deg: int, amplitude: float=6):
        doas = PositionGenerator.fixed_spread_doas(num, spread_deg)
        return np.stack([np.cos(doas), np.sin(doas)], axis=1)*amplitude

    @staticmethod
    def doas_to_positions(doas: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
        return (np.stack([np.cos(doas), np.sin(doas)], axis=1).T*amplitudes).T

class SignalGenerator:
    """Helper class to create source signals"""
    @staticmethod
    def sawtooths(n_sources: int, sampling_frequency: int, duration: int) -> list[np.ndarray]:
        t = np.linspace(0, 1, num=int(sampling_frequency * duration))
        return np.array(
            [
                (scipy.signal.sawtooth(t * j * 400 * 2 * np.pi)).astype(np.float32)
                for j in range(1, n_sources + 1)
            ]
        )
    
    @staticmethod
    def car_exhausts(n_sources: int, sampling_frequency: int, duration: int) -> np.ndarray:
        def single_car_exhaust(
            low_freq: int,
            med_freq: int, 
            high_freq: int,
            general_modulation=1.0,
            duration: int=2,
            sampling_frequency=44100,
        ):
            t = np.linspace(0, duration, num=int(sampling_frequency * duration))
            twopi = 2 * np.pi
            samples = (
                np.sin(t * low_freq * twopi)
                + np.sin(t * med_freq * twopi)
                + 0.5 * np.sin(t * high_freq * twopi)
            )
            samples *= np.sin(general_modulation * t * twopi)
            samples = samples.astype(np.float32)
            return samples
        
        return np.array([
            single_car_exhaust(
                np.random.randint(20, 80),
                np.random.randint(100, 200),
                np.random.randint(250, 400),
                general_modulation=np.random.randint(20, 40),
                duration=duration,
                sampling_frequency=sampling_frequency,
            )
            for _ in range(n_sources)
        ])

def play(samples: np.ndarray, sampling_frequency: int):
    """Play out a sound using pyaudio"""
    output_bytes = samples.tobytes()
    p = pyaudio.PyAudio()
    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=sampling_frequency, output=True)

    start_time = time.time()
    stream.write(output_bytes)
    log.info("Played sound for {:.2f} seconds".format(time.time() - start_time))

    stream.stop_stream()
    stream.close()
    p.terminate()