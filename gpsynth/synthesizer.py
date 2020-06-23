import datetime
import json
import os
import random
from typing import Union, List, Optional

import GPy
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

import gpsynth.config as config
from gpsynth.audio_output import WavFile, RealtimeAudio


def midi_to_frequency(midi_note: Union[float, int]) -> float:
    """Converts MIDI note number to frequency in Hz.

    :param midi_note: The MIDI note.
    :return: The frequency in Hz.
    """
    half_tone = 2 ** (1 / 12)
    return 440. * half_tone ** (midi_note - 69.)


def make_cov_cholesky_waveshaping(kernel: GPy.kern.Kern) -> np.ndarray:
    """Compute the Cholesky decomposition for waveshaping synthesis.

    :param kernel: The GP kernel
    :return: The Cholesky decomposition
    """
    #  Remark: Since we are doing waveshaping, it is not necessary to consider
    #  periodic / non-periodic kernels separately.
    samples = 44100 / 20
    xs = np.arange(samples) * 2. * np.pi / samples
    xs = np.sin(xs)
    cov = kernel.K(xs[:, None], xs[:, None])
    chol = GPy.util.linalg.jitchol(cov)
    return chol


def make_cov_cholesky(kernel: GPy.kern.Kern) -> np.ndarray:
    """Compute the Cholesky decomposition for waveshaping synthesis.

    :param kernel: The GP kernel
    :return: The Cholesky decomposition
    """
    #  Remark: Since we are doing wavetable synthesis, it is necessary to
    #  consider periodic / non-periodic kernels separately in order to ensure
    #  good continuation.
    samples = 44100 / 20
    xs = np.arange(samples + 1) * 2. * np.pi / samples
    if isinstance(kernel, GPy.kern.PeriodicExponential.__bases__[0]) or not config.good_continuation_regression:
        # print('Is periodic')
        X = np.array([xs[0]])[:, None]
        Y = np.array([0.])[:, None]
    else:
        X = np.array([xs[0], xs[-1]])[:, None]
        Y = np.array([0., 0.])[:, None]
    m = GPy.models.GPRegression(X, Y, kernel)
    m.Gaussian_noise = 0.0
    mean, cov = m.predict_noiseless(xs[:, None], full_cov=True)
    chol = GPy.util.linalg.jitchol(cov)
    return chol


def perceptual_amplitude_dbb(frequency: float) -> float:
    """Perceptual amplitude according to db(B).

    :param frequency: Frequency in Hz.
    :return: An amplitude to scale the given frequency with.
    """
    # See http://www.sengpielaudio.com/BerechnungDerBewertungsfilter.pdf

    num = 12200.0 ** 2. * frequency ** 3
    den = (frequency ** 2. + 20.6) * (frequency ** 2. + 12200. ** 2.) * np.sqrt(frequency ** 2. + 158.5 ** 2.)
    return num / den


def fast_normal_from_cholesky(cholesky: np.ndarray) -> np.ndarray:
    """Efficiently samples a multidimensional normal from the Cholesky
    decomposition of the covariance matrix.

    :param cholesky: The Cholesky decomposition.
    :return: A sample of the mutidimensional normal distribution.
    """
    n = cholesky.shape[0]
    means = np.zeros(n)
    while True:
        seeds = np.random.normal(0, 1, (1, n))
        Ls = [cholesky]
        result = means + np.einsum('nij,njk->nik', Ls, seeds[:, :, np.newaxis])[:, :, 0]
        result = result - np.mean(result)
        result = result / np.std(result) / 10.0

        good_loudness = 300.
        actual_loudness = weighted_loudness(result[0], mult_freq=263. / 20.)
        result = result / actual_loudness * good_loudness

        if np.max(np.abs(result)) < 0.9:
            break

    return result


class GPSynth:
    def __init__(self, kernel: GPy.kern.Kern, out_rt: Optional[RealtimeAudio], out_wav: Optional[WavFile],
                 n_wavetables: int = 17, waveshaping: bool = False):
        """GPSynth creates wavetables based on a kernel of a Gaussian Process.

        :param kernel: The kernel.
        :param out_rt: Is used for real-time audio output if not None.
        :param out_wav: Is used for saving the output to a WAV file if not None.
        :param n_wavetables: The number of (randomized) wavetables to be generated.
        :param waveshaping: Should waveshaping be used?
        """
        self.table_idx = 0
        self.wavetables = make_wavetables(kernel, n_wavetables, waveshaping)
        self.out_rt = out_rt
        self.out_wav = out_wav

    def note(self, midi_note: Union[int, float], duration: float) -> None:
        """Plays a note.

        :param midi_note: The MIDI pitch of the note.
        :param duration: The duration.
        """

        wavetable = self.wavetables[self.table_idx]

        size_wavetable = wavetable.size
        w = np.concatenate((wavetable, wavetable, wavetable))

        fs = 44100.
        fc = 20000. * 20. / midi_to_frequency(midi_note)  # cutoff frequency
        fc_norm = fc / (fs / 2)
        b, a = signal.butter(5, fc_norm)
        y = signal.filtfilt(b, a, w)

        wavetable = y[size_wavetable:2 * size_wavetable]  # the middle part

        pointer_idx = 0.0
        step = midi_to_frequency(midi_note) / 44100.0 * wavetable.shape[0]
        samples_total = int(duration * 44100.)
        pcm = np.zeros((samples_total), dtype=np.float32)
        for i in range(samples_total):
            pcm[i] = wavetable[int(pointer_idx)]
            fade_in = 100
            if i < fade_in:
                pcm[i] *= i / fade_in

            fade_out = 10000
            if samples_total - i < fade_out:
                pcm[i] *= (samples_total - i) / fade_out

            pointer_idx += step
            if pointer_idx >= wavetable.shape[0]:
                pointer_idx -= wavetable.shape[0]

        self.table_idx = (self.table_idx + 1) % len(self.wavetables)

        if self.out_rt is not None:
            self.out_rt.write_samples(pcm)
        if self.out_wav is not None:
            self.out_wav.write_samples(pcm)

    def save_wavetables(self, path: str, filename_prefix: str = '') -> None:
        """Saves the generated wavetables.

        :param path: The path, the wavetables should be saved to.
        :param filename_prefix: The prefix of the filename.
        """
        for i in range(len(self.wavetables)):
            if not os.path.exists(path):
                os.mkdir(path)
            location = os.path.join(path, filename_prefix + f'{i:02d}.wav')
            wav_file = WavFile(location)
            wav_file.write_samples(self.wavetables[i])


def kernel_for_string(name: str, lengthscale: float = 1.) -> GPy.kern.Kern:
    """Convenience function to make a kernel.

    :param name: The name of the kernel.
    :param lengthscale: The length-scale parameter
    :return: The kernel.
    """
    variance = .3 ** 2
    if name == 'RBF':
        return GPy.kern.RBF(input_dim=1, lengthscale=lengthscale)
    if name == 'Exponential':
        return GPy.kern.Exponential(input_dim=1, lengthscale=lengthscale)
    if name == 'Matern32':
        return GPy.kern.Matern32(input_dim=1, lengthscale=lengthscale)
    if name == 'Matern52':
        return GPy.kern.Matern52(input_dim=1, lengthscale=lengthscale)
    if name == 'PeriodicExponential':
        return GPy.kern.PeriodicExponential(input_dim=1, period=2. * np.pi, lengthscale=lengthscale, variance=variance)
    if name == 'PeriodicMatern32':
        return GPy.kern.PeriodicMatern32(input_dim=1, period=2. * np.pi, lengthscale=lengthscale, variance=variance)
    if name == 'PeriodicMatern52':
        return GPy.kern.PeriodicMatern52(input_dim=1, period=2. * np.pi, lengthscale=lengthscale)
    if name == 'StdPeriodic':
        return GPy.kern.StdPeriodic(input_dim=1, period=2. * np.pi, lengthscale=lengthscale)
    if name == 'Brownian':
        return GPy.kern.Brownian(input_dim=1)
    if name == 'ExpQuad':
        return GPy.kern.ExpQuad(input_dim=1, lengthscale=lengthscale)
    if name == 'OU':
        return GPy.kern.OU(input_dim=1, lengthscale=lengthscale)
    if name == 'RatQuad':
        return GPy.kern.RatQuad(input_dim=1, lengthscale=lengthscale)
    if name == 'White':
        return GPy.kern.White(input_dim=1)
    if name == 'MLP':
        return GPy.kern.MLP(input_dim=1)  # has other parameters
    if name == 'Spline':
        return GPy.kern.Spline(input_dim=1)
    if name == 'Poly':
        return GPy.kern.Poly(input_dim=1)  # has other parameters

    raise LookupError()


def make_wavetables(kernel: GPy.kern.Kern, n: int = 17, waveshaping: bool = False) -> List[np.ndarray]:
    """Generates wavetables from kernel.

    :param kernel: The kernel.
    :param n: The number of wavetables to be generated.
    :param waveshaping: Should waveshaping be used.
    :return: A list of wavetables.
    """
    wavetables = []

    if not waveshaping:
        cholesky = make_cov_cholesky(kernel)
    else:
        cholesky = make_cov_cholesky_waveshaping(kernel)
    for _ in range(n):
        wavetable = fast_normal_from_cholesky(cholesky)[0]
        wavetables.append(wavetable[:-1])

    return wavetables


def plot_spectrum(wavetable: np.ndarray) -> None:
    """Plots the spectrum of the wavetable.

    :param wavetable: The wavetable.
    """
    ps = np.abs(np.fft.fft(wavetable)) ** 2

    time_step = 1 / 44100
    freqs = np.fft.fftfreq(wavetable.size, time_step)
    idx = np.argsort(freqs)

    plt.plot(freqs[idx], ps[idx])
    plt.show()


def weighted_loudness(wavetable: np.ndarray, mult_freq: float = 1.):
    """Calculates the perceived loudness according to db(B) of a note played
    with the wavetable.

    :param wavetable: The wavetable.
    :param mult_freq: The frequency of the note as a multiple of 20 Hz.
    :return: The perceived loudness.
    """
    ps = np.abs(np.fft.fft(wavetable))

    time_step = 1 / 44100
    freqs = np.fft.fftfreq(wavetable.size, time_step)
    idx = np.argsort(freqs)

    weighted_sum = 0
    for i in idx:
        freq = freqs[i]
        if freq > 0:
            weighted_sum += perceptual_amplitude_dbb(freq * mult_freq) * ps[i]

    return weighted_sum


def big_sweep(all_kernels: List[GPy.kern.Kern], path: str, ls_subdivisions: int = 16, n_wavetables: int = 7) -> None:
    """Creates wavetables for all kernels with different length scales with
    multiplicative and additive combinations. The result can be used for sound
    synthesis (for example in pureData, SuperCollider or Max/MSP.

    :param all_kernels: The list of all kernels.
    :param path: The path where the wavetables are stored.
    :param ls_subdivisions: Number of length-scale subdivisions.
    :param n_wavetables: The number of (randomized) wavetables per setting.
    """
    out_long = WavFile(os.path.join(path, 'c.wav'))

    delta_t = 1.
    ls_start = 0.01
    ls_end = np.pi

    score = []
    time = 0.
    l_vals = np.geomspace(ls_start, ls_end, ls_subdivisions).tolist()

    n_combinations = 1000
    for _ in range(n_combinations):
        k1_str = random.choice(all_kernels)
        while True:
            k2_str = random.choice(all_kernels)
            if k2_str != k1_str:
                break
        l1 = random.choice(l_vals)
        l2 = random.choice(l_vals)
        l1_idx = l_vals.index(l1)
        l2_idx = l_vals.index(l2)

        k1 = kernel_for_string(k1_str, lengthscale=l1)
        k2 = kernel_for_string(k2_str, lengthscale=l2)
        operator = random.choice(['plus', 'times'])
        if operator == 'plus':
            kernel = k1 + k2
        else:
            kernel = k1 * k2

        waveshaping = random.choice([True, False])

        synth = GPSynth(kernel, out_rt=None, out_wav=out_long, n_wavetables=n_wavetables, waveshaping=waveshaping)
        print(f'waveshaping={waveshaping}', k1_str, l1, operator, k2_str, l2)
        for n_idx in range(1):  # only one note to c.wav otherwise the file becomes too big for the web.
            score.append({
                'kernel_1': k1_str,
                'operator': 'plus',
                'kernel_2': k2_str,
                'lengthscale_1': l1,
                'lengthscale_1_idx': l1_idx,
                'lengthscale_2': l2,
                'lengthscale_2_idx': l2_idx,
                'waveshaping': waveshaping,
                'time': time,
                'note': n_idx
            })
            synth.note(60, delta_t)
            time += delta_t

        waveshaping_str = 'waveshaping_' if waveshaping else ''
        prefix = waveshaping_str + k1_str + f'_l{l1_idx:03d}(plus)' + k2_str + f'_l{l2_idx:03d}_n'
        synth.save_wavetables(os.path.join(path, 'samples'), prefix)

    for waveshaping in [False, True]:
        for kernel_str in all_kernels:
            ls_start = 0.01
            ls_end = np.pi
            l_vals = np.geomspace(ls_start, ls_end, ls_subdivisions)
            for l_idx, lengthscale in enumerate(l_vals):
                k = kernel_for_string(kernel_str, lengthscale=lengthscale)
                synth = GPSynth(k, out_rt=None, out_wav=out_long, n_wavetables=n_wavetables, waveshaping=waveshaping)
                print(f'waveshaping={waveshaping}', kernel_str, lengthscale, f'waveshaping = {waveshaping}')
                for n_idx in range(1):  # only one note to c.wav otherwise the file becomes too big for the web.
                    score.append({
                        'kernel_1': kernel_str,
                        'operator': '',
                        'kernel_2': '',
                        'lengthscale_1': lengthscale,
                        'lengthscale_1_idx': l_idx,
                        'lengthscale_2': -1,
                        'lengthscale_2_idx': -1,
                        'waveshaping': waveshaping,
                        'time': time,
                        'note': n_idx
                    })
                    synth.note(60, delta_t)
                    time += delta_t

                waveshaping_str = 'waveshaping_' if waveshaping else ''
                prefix = waveshaping_str + kernel_str + f'_l{l_idx:03d}_n'
                synth.save_wavetables(os.path.join(path, 'samples'), prefix)

    with open(os.path.join(path, 'score.json'), 'w') as f:
        json.dump(score, f, indent=4)


all_kernels = [
    'RBF', 'Exponential', 'Matern32', 'Matern52', 'PeriodicExponential', 'PeriodicMatern32', 'PeriodicMatern52',
    'StdPeriodic', 'ExpQuad', 'OU', 'RatQuad', 'MLP', 'Spline', 'Poly'
]


def main():
    dir_name = datetime.datetime.now().strftime('%Y%m%d-%H%M') + '_multiexport'
    path = os.path.join('..', 'results', dir_name)
    os.makedirs(path, exist_ok=True)
    big_sweep(all_kernels, path)


if __name__ == '__main__':
    main()
