import wave
import struct
import datetime
import math

import pyaudio
import numpy as np


class RealtimeAudio:
    """Real-time audio output"""

    def __init__(self):
        self.pyaudio = pyaudio.PyAudio()
        fs = 44100  # sampling rate, Hz, must be integer
        self.stream = self.pyaudio.open(format=pyaudio.paInt16, channels=1, rate=fs, output=True)

    def write_samples(self, samples: np.ndarray) -> None:
        """Plays the samples immediately.

        :param samples: The samples should be between -1. and 1.
        :return: None
        """

        samples = (samples * (2 ** 15)).astype(np.int16)
        self.stream.write(samples.tostring())

    def close(self):
        """Stops the audio stream."""

        self.stream.stop_stream()
        self.stream.close()
        self.pyaudio.terminate()


class WavFile:
    """Save output to WAV file"""

    def __init__(self, path: str):
        """Prepares to save the audio as WAV file.

        :param path: Path where the WAV file is created.
        """

        self.wav_file = wave.open(path, 'w')
        self.wav_file.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))  # mono, 16 bits, 44100 Hz

    def close(self):
        """Closes the WAV file."""

        self.wav_file.close()

    def write_samples(self, samples: np.ndarray) -> None:  # sample is a float in the range [-1, +1]
        """Writes the sample into the WAV file.

        :param samples: The samples should be between -1. and 1.
        :return: None
        """

        for i in range(samples.size):
            sample = samples[i]
            audio_sample = int(sample * (math.pow(2, 15) - 1))
            self.wav_file.writeframesraw(struct.pack("<h", audio_sample))


def main():
    fs = 44100
    duration = 0.1
    f = 440.0
    samples = (np.sin(2 * np.pi * np.arange(fs * duration) * f / fs)).astype(np.float32)

    rt_out = RealtimeAudio()
    rt_out.write_samples(samples)
    rt_out.close()

    filename = datetime.datetime.now().strftime('%m%d-%H%M') + '.wav'
    wav_out = WavFile(filename)
    wav_out.write_samples(samples)
    wav_out.close()


if __name__ == '__main__':
    main()
