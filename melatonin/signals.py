import time
from scipy.special import expit
import numpy as np
import pyaudio


def car_exhaust(
    frequencies,
    revving_zone,
    general_modulation=1.0,
    duration=2.0,
    volume=0.4,
    fs=44100,
):
    t = np.linspace(0, duration, num=int(fs * duration))
    samples = (
        np.sin(t * frequencies[0] * 2 * np.pi)
        + np.sin(
            (
                t * frequencies[1]
                + expit((t - revving_zone[0]) * 7 - 6)
                * frequencies[2]
                * (t < revving_zone[1])
                * (t >= revving_zone[0])
                # + expit(t*5 - 6) * 2000 * (t>=1)
            )
            * 2
            * np.pi
        )
        + 0.5
        * np.sin(
            (
                t * frequencies[3]
                + expit((t - revving_zone[0]) * 7 - 6)
                * frequencies[4]
                * 2
                * np.pi
                * (t < revving_zone[1])
                * (t >= revving_zone[0])
            )
        )
    )
    samples *= np.sin(general_modulation * t * 2 * np.pi)
    samples = (samples).astype(np.float32)
    return t, volume * samples


def play(samples, fs):
    output_bytes = samples.tobytes()
    p = pyaudio.PyAudio()
    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)

    start_time = time.time()
    stream.write(output_bytes)
    print("Played sound for {:.2f} seconds".format(time.time() - start_time))

    stream.stop_stream()
    stream.close()

    p.terminate()
