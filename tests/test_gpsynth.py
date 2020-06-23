import pytest
import datetime
import os

import numpy as np

from gpsynth.audio_output import WavFile, RealtimeAudio
from gpsynth.synthesizer import GPSynth, kernel_for_string, all_kernels


def test_audio_output(tmp_path: str):
    fs = 44100
    duration = 0.1
    f = 440.0
    samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

    rt_out = RealtimeAudio()
    rt_out.write_samples(samples)
    rt_out.close()

    path = os.path.join(tmp_path, 'result.wav')
    wav_out = WavFile(path)
    wav_out.write_samples(samples)
    wav_out.close()

    assert os.path.isfile(path)


def test_synthesizer(tmp_path: str):
    rta = RealtimeAudio()
    wav = WavFile(os.path.join(tmp_path, 'test_gpsynth.wav'))
    for idx, k_str in enumerate(all_kernels):
        kernel = kernel_for_string(k_str, lengthscale=1.)
        for waveshaping in [True, False]:
            synth = GPSynth(kernel, rta, wav, 3, waveshaping)
            synth.note(60., 0.1)
            synth.save_wavetables(tmp_path, f'{idx}{waveshaping}.wav')
