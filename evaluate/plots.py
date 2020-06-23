import os

import numpy as np
import GPy
from matplotlib import pyplot as plt
from scipy.io import wavfile

import evaluate

colors = ['g', 'k', 'b']
linestyles = ['-', ':', '-.', '--']


def plot_samples(k: GPy.kern.Kern, directory: str) -> None:
    """Plots samples from a kernel.

    :param k: The kernel from which to draw samples
    :return: None
    """

    X = np.linspace(0., 10., 500)
    X = X[:, None]
    mu = np.zeros(500)
    C = k.K(X, X)
    Z = np.random.multivariate_normal(mu, C, 20)
    fig, ax = plt.subplots(1, 1)

    for i in range(3):
        c = colors[i]
        ls = linestyles[i]
        ax.plot(X[:], Z[i, :], color=c, linestyle=ls)

    ls = f'l{k.lengthscale[0]}'.replace('.', '_')
    path = os.path.join(directory, f"samples_{k.name}_{ls}.pdf")
    plt.savefig(path, bbox_inches='tight')


def plot_samples_with_regression(k: GPy.kern, directory: str) -> None:
    """Plots samples forced to pass through the same point at the center.

    :param k: The kernel from which to draw samples
    :return: None
    """

    X = np.linspace(0., 10., 500)
    X = X[:, None]
    X_regress = np.array([X[250]])
    Y_regress = np.array([0.])[:, None]
    m = GPy.models.GPRegression(X_regress, Y_regress, k)
    m.Gaussian_noise = 0.0
    mean, cov = m.predict_noiseless(X, full_cov=True)
    mean = mean[:, 0]

    Z = np.random.multivariate_normal(mean, cov, 20)

    fig, ax = plt.subplots(1, 1)

    for i in range(3):
        c = colors[i]
        ls = linestyles[i]
        ax.plot(X[:], Z[i, :], color=c, linestyle=ls)

    ls = f'l{k.lengthscale[0]}'.replace('.', '_')
    path = os.path.join(directory, f"regression_{k.name}_{ls}.pdf")
    plt.savefig(path, bbox_inches='tight')


def plot_spectrogram(wav_path: str, out_path: str) -> None:
    """Plots the magnitude spectrogram.

    :param wav_path: Path to the WAV file.
    :param out_path: Path of the output file.
    :return: None
    """
    samplingFrequency, signalData = wavfile.read(wav_path)
    fig = plt.figure()
    pxx, freq, t, cax = \
        plt.specgram(signalData, Fs=samplingFrequency, mode='magnitude', scale='dB', NFFT=1024, noverlap=512)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    fig.colorbar(cax).set_label('Magnitude [dB]')
    fig.tight_layout()

    fig.savefig(out_path)


def main(directory: str):
    k = GPy.kern.RBF(input_dim=1, lengthscale=1.)
    plot_samples(k, directory)
    k = GPy.kern.Exponential(input_dim=1, lengthscale=1.)
    plot_samples(k, directory)
    k = GPy.kern.Matern32(input_dim=1, lengthscale=1.)
    plot_samples(k, directory)
    k = GPy.kern.Matern52(input_dim=1, lengthscale=1.)
    plot_samples(k, directory)

    k = GPy.kern.RBF(input_dim=1, lengthscale=0.2)
    plot_samples(k, directory)
    k = GPy.kern.RBF(input_dim=1, lengthscale=1.0)
    plot_samples(k, directory)
    k = GPy.kern.RBF(input_dim=1, lengthscale=5.0)
    plot_samples(k, directory)

    plot_samples_with_regression(k, directory)

    wav_path_good = os.path.join(directory, 'good_continuation=True.wav')
    wav_path_bad = os.path.join(directory, 'good_continuation=False.wav')
    if not os.path.exists(wav_path_good):
        assert not os.path.exists(wav_path_bad)  # don't overwrite existing file
        evaluate.make_continuous_discontinuous(directory)  # generates the WAV files

    plot_spectrogram(wav_path_good, os.path.join(directory, 'good_continuation.pdf'))
    plot_spectrogram(wav_path_bad, os.path.join(directory, 'bad_continuation.pdf'))


if __name__ == '__main__':
    if not os.path.exists('../results'):
        os.mkdir('../results')
    main('../results')
