from dataclasses import dataclass
import scipy.fft
import numpy as np
import scipy
from matplotlib import pyplot as plt


@dataclass
class Parameters:
    microphone_positions: np.ndarray
    slice_size: int = 2048
    fs: int = 44100
    adjacent_zone: int = 2
    single_source_threshold: float = 0.8
    speed_of_sound: int = 1
    # t = 10
    D: int = 4  # detection
    L: int = 50  # histogram length
    Q0: int = 5  # blackman window peak
    max_sources: int = 10
    verbose: bool = False

    @property
    def overlap_size(self):
        return self.slice_size // 2

    @property
    def Q(self):
        return 2 * self.Q0 + 1

    @property
    def N_mics(self):
        return self.microphone_positions.shape[0]

    @property
    def angle_rotation(self):
        return np.pi * 2 / self.N_mics

    @property
    def A(self):
        return np.pi / 2 + self.angle_rotation / 2

    @property
    def A_prime(self):
        return self.A + np.pi / 2

    @property
    def distance_to_next_mic(self):
        return np.linalg.norm(
            self.microphone_positions[0, :] - self.microphone_positions[1, :]
        )


class Detector:
    def __init__(self, parameters: Parameters, mic_signals):
        self.parameters = parameters
        self.mic_signals = mic_signals
        self.mic_time_slices = []

    def detect(self, t_from, t_to):
        self.mic_time_slices = []

        for mic_i, signal in enumerate(self.mic_signals):
            self.mic_time_slices.append(list())
            for j in range(len(signal) // self.parameters.slice_size):
                overlapping_mask = self.parameters.overlap_size * (j > 0)
                self.mic_time_slices[mic_i].append(
                    signal[
                        j * self.parameters.slice_size
                        - overlapping_mask : (j + 1) * self.parameters.slice_size
                        - overlapping_mask
                    ]
                )
        # [Since slice_size = 2048 then the FFT is also 2048]
        self.mic_fft_slices = [
            [scipy.fft.rfft(_slice) for _slice in slices]
            for slices in self.mic_time_slices
        ]

        # We then define a “constant-time analysis zone”, (t, Ω), as a
        # series of frequency-adjacent TF points (t, ω). A “constant-time
        # analysis zone”, (t, Ω) is thus referred to a specific time frame t
        # and is comprised by Ω adjacent frequency components.

        self.freq_bins = scipy.fft.rfftfreq(
            self.parameters.slice_size, 1 / self.parameters.fs
        )
        if self.parameters.verbose:
            print(
                f"Single source analysis zone is {self.freq_bins[self.parameters.adjacent_zone+1] - self.freq_bins[1]:.2f}Hz"
            )

        # Choosing the important frequencies
        # The paper selects w_i_max for each microphone pair for each single source zone.
        self.doa_zone_estimations = []
        self.frequencies_of_interest = []
        for t in range(t_from, t_to):
            for zone in range(50):
                if not self.is_single_source_zone(zone, t, self.mic_fft_slices):
                    continue

                top_frequency_indices = self.d_highest_peaks(
                    zone * self.parameters.adjacent_zone,
                    (zone + 1) * self.parameters.adjacent_zone,
                    timestep=t,
                    d=self.parameters.D,  # TODO: move this to function?
                    mic_fft_slices=self.mic_fft_slices,
                )

                for frequency_index in top_frequency_indices:
                    # TODO: checkme - to avoid spurious DoA estimations
                    if np.abs(self.mic_fft_slices[1][t][frequency_index]) < 100:
                        continue
                    print(
                        f"Using frequency {self.freq_bins[frequency_index]} in zone #{zone}"
                    )
                    self.frequencies_of_interest.append(self.freq_bins[frequency_index])
                    # print(f"Value: {np.abs(self.mic_fft_slices[1][t][frequency_index])}")
                    # for single source zone, detect DoA
                    result = scipy.optimize.minimize_scalar(
                        self.negative_cics,
                        method="bounded",
                        bounds=(0, 2 * np.pi),
                        args=(frequency_index, t, self.mic_fft_slices, self.freq_bins),
                    )
                    self.doa_zone_estimations.append(result.x)

        self.bins, self.x = np.histogram(
            np.array(self.doa_zone_estimations) * 360 / (2 * np.pi),
            bins=np.linspace(0, 360, self.parameters.L + 1),
        )

        # Source atom detection
        window = scipy.signal.windows.blackman(self.parameters.Q)
        u = np.zeros(self.parameters.L)
        u[: self.parameters.Q] = window
        c = np.roll(u, -self.parameters.Q0)
        C = np.array([np.roll(c, k) for k in range(self.parameters.L)])

        current_histogram = self.bins
        self.atoms = []
        self.atom_energies = []
        self.window_energy = np.dot(c, c)
        self.energy_threshold = self.bins.mean()
        self.atom_contributions = []
        for j in range(self.parameters.max_sources):
            # We skip the further-than condition from step 3 of the counting algorithm (P5)
            # because it should be solved by the contribution step.
            corr_window = C.dot(current_histogram)
            position = np.argmax(corr_window)
            atom_energy = corr_window.max() / self.window_energy
            if atom_energy < self.energy_threshold:
                break
            atom_contribution = C[position, :] * atom_energy
            current_histogram = current_histogram - atom_contribution
            self.atoms.append(position)
            self.atom_energies.append(atom_energy)
            self.atom_contributions.append(atom_contribution)

    def d_highest_peaks(self, freq_from, freq_to, timestep, d, mic_fft_slices):
        magnitudes = {}
        for freq in range(freq_from, freq_to):
            val = 0
            for i in range(self.parameters.N_mics):
                next_mic = (i + 1) % self.parameters.N_mics
                val += np.abs(
                    mic_fft_slices[i][timestep][freq]
                    * np.conj(mic_fft_slices[next_mic][timestep][freq])
                )
            magnitudes[freq] = val
        return sorted(magnitudes, key=magnitudes.__getitem__)[-2:]

    def correlation(self, mic1, mic2, timestep, f_from, f_to, mic_fft_slices):
        return np.linalg.norm(
            mic_fft_slices[mic1][timestep][f_from:f_to]
            * mic_fft_slices[mic2][timestep][f_from:f_to],
            ord=1,
        )

    def cross_correlation(self, mic1, mic2, timestep, f_from, f_to, mic_fft_slices):
        # Cross correlation
        corr = self.correlation(mic1, mic2, timestep, f_from, f_to, mic_fft_slices)

        # Correlation coefficient
        correlation_coefficient = corr / np.sqrt(
            self.correlation(
                mic1,
                mic1,
                timestep=timestep,
                f_from=f_from,
                f_to=f_to,
                mic_fft_slices=mic_fft_slices,
            )
            * self.correlation(
                mic2,
                mic2,
                timestep=timestep,
                f_from=f_from,
                f_to=f_to,
                mic_fft_slices=mic_fft_slices,
            )
        )
        return correlation_coefficient

    def is_single_source_zone(self, zone_index, timestep, mic_fft_slices):
        avg = 0
        omega_index = self.parameters.adjacent_zone * zone_index
        for i in range(1, self.parameters.N_mics):
            next_mic = (i + 1) % self.parameters.N_mics
            avg += (
                1
                / self.parameters.N_mics
                * self.cross_correlation(
                    i,
                    next_mic,
                    timestep,
                    omega_index,
                    omega_index + self.parameters.adjacent_zone,
                    mic_fft_slices,
                )
            )
        return avg >= self.parameters.single_source_threshold

    def negative_cics(self, phi, omega_index, t, mic_fft_slices, freq_bins):
        """Circular Integrated Cross Spectrum"""
        value = 0
        omega = freq_bins[omega_index]
        for i in range(self.parameters.N_mics):
            # +1 because we use zero-index; eqn uses -1
            phase_rotation_factor = np.exp(
                -1j
                * omega
                * self.parameters.distance_to_next_mic
                / self.parameters.speed_of_sound
                * (
                    np.sin(self.parameters.A_prime - phi)
                    - np.sin(
                        self.parameters.A_prime
                        - phi
                        + (i + 1 - 1) * self.parameters.angle_rotation
                    )
                )
            )
            # What happens in the last microphone?? I'm guessing wrap-around
            cross_power = mic_fft_slices[i][t][omega_index] * np.conj(
                mic_fft_slices[((i + 1) % self.parameters.N_mics)][t][omega_index]
            )

            phase_cross_spectrum = cross_power / np.conj(cross_power)
            value += phase_cross_spectrum * phase_rotation_factor
        return -np.abs(value)

    def doa_estimation_histogram(self):
        plt.figure(dpi=150, figsize=(5, 3))
        plt.title(
            f"Histogram of DOA estimations\nusing {self.parameters.D} frequency components"
        )

        bins, x, _ = plt.hist(
            np.array(self.doa_zone_estimations) * 360 / (2 * np.pi),
            bins=np.linspace(0, 360, self.parameters.L + 1),
        )
        plt.xlabel("Direction of Arrival degrees")
        plt.ylabel("# Estimation")

    def scatter_doa_estimation(self):
        plt.figure()

        bins, x = np.histogram(
            np.array(self.doa_zone_estimations) * 360 / (2 * np.pi),
            bins=np.linspace(0, 360, self.parameters.L + 1),
        )

        doa_hist = dict(zip(x, bins))

        plt.scatter(
            self.parameters.microphone_positions[:, 0],
            self.parameters.microphone_positions[:, 1],
            label="mic",
            color="blue",
            marker="o",
        )
        max_value = bins.max()
        for value, freq in doa_hist.items():
            plt.plot(
                [0, 10 * np.sin(value / 360 * 2 * np.pi - np.pi)],
                [0, 10 * np.cos(value / 360 * 2 * np.pi - np.pi)],
                alpha=freq / max_value,
                color="black",
                linestyle="-.",
            )
        plt.ylim(-12, 12)
        plt.xlim(-12, 12)
        plt.xticks(np.linspace(-12, 12, 25))
        plt.yticks(np.linspace(-12, 12, 25))
        plt.legend()
        plt.grid(alpha=0.5)
        plt.title("DoA estimation")

    def atom_detection_histogram(self):
        plt.figure(dpi=150, facecolor="white", figsize=(5, 3))
        plt.plot(np.arange(self.parameters.L), self.bins, label="Original histogram")
        for j, atom_contribution in enumerate(self.atom_contributions):
            plt.plot(
                np.arange(self.parameters.L),
                atom_contribution,
                label=f"Est. source #{j}",
            )
        plt.title("Atom detection on histogram")
        plt.grid()
        plt.legend()

    def plot_predictions(self):
        plt.figure(dpi=150, facecolor="white", figsize=(5, 3))
        plt.scatter(
            self.parameters.microphone_positions[:, 0],
            self.parameters.microphone_positions[:, 1],
            label="mic",
            color="blue",
            marker="o",
        )
        for atom, energy in zip(self.atoms, self.atom_energies):
            value = self.x[atom]
            plt.plot(
                [0, 10 * np.sin(value / 360 * 2 * np.pi)],
                [0, 10 * np.cos(value / 360 * 2 * np.pi)],
                color="gray",
                alpha=energy / max(self.atom_energies),
            )

        plt.title("Predictions")
        plt.grid()
        plt.legend()
