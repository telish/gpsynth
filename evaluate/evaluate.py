import os
import time

import GPy
import numpy as np
from tqdm import tqdm

import gpsynth.config as config
from gpsynth.audio_output import WavFile, RealtimeAudio
from gpsynth.synthesizer import GPSynth, kernel_for_string


def make_continuous_discontinuous(directory: str) -> None:
    """Create WAV files with and without enforcing continuity.

    :param directory: Directory where the results are saved.
    :return: None
    """

    for setting in [True, False]:
        print('Good continuation:', setting)
        config.GOOD_CONTINUATION_REGRESSION = setting
        path = os.path.join(directory, f'good_continuation={config.GOOD_CONTINUATION_REGRESSION}.wav')
        out_wav = WavFile(path)
        ls_start = 0.01
        ls_end = 1.
        l_vals = np.geomspace(ls_start, ls_end, 10)
        for l_idx, lengthscale in enumerate(l_vals):
            k = kernel_for_string('RBF', lengthscale=lengthscale)
            print('l =', lengthscale)
            gpsynth = GPSynth(k, None, out_wav, n_wavetables=1)
            gpsynth.note(60, 1.)


def speed_test(n_wavetables=1000):
    """How long does it take to compute the wavetables."""

    start = time.time()
    kernel = GPy.kern.RBF(input_dim=1, lengthscale=1.0)
    gpsynth = GPSynth(kernel, out_rt=None, out_wav=None, n_wavetables=n_wavetables, waveshaping=False)
    end = time.time()
    print('Time elapsed (saved Cholesky decomposition)', end - start)

    start = time.time()
    for i in tqdm(range(n_wavetables)):
        kernel = GPy.kern.RBF(input_dim=1, lengthscale=1.0)
        gpsynth = GPSynth(kernel, out_rt=None, out_wav=None, n_wavetables=1, waveshaping=False)
    end = time.time()
    print('Time elapsed (full computation)', end - start)


def main(directory: str):
    make_continuous_discontinuous(directory)
    speed_test()


if __name__ == '__main__':
    if not os.path.exists('../results'):
        os.mkdir('../results')
    main('../results')
