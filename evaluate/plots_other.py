import numpy as np
from matplotlib import pyplot as plt

from gpsynth.synthesizer import kernel_for_string


def plot_kernel(k):
    plt.figure(figsize=(18, 5))
    X = np.linspace(-4., 4., 1000)[:, None]  # we need [:, None] to reshape X into a column vector for use in GPy

    plt.subplot(121)
    K = k.K(X, np.array([[0.]]))
    plt.plot(X, K)
    plt.xlabel("x"), plt.ylabel("$\kappa$")

    X = np.linspace(-4., 4., 100)[:, None]  # we need [:, None] to reshape X into a column vector for use in GPy
    plt.subplot(122)
    K = k.K(X, X)
    plt.pcolor(X.T, X, K)
    plt.gca().invert_yaxis(), plt.gca().axis("image")
    plt.xlabel("x"), plt.ylabel("x'"), plt.colorbar()
    plt.title("$\kappa_{rbf}(x,x')$")
    plt.show()


def plot_variance(k):
    X = np.linspace(-4., 4., 1000)[:, None]
    # List of variances
    vs = [0.1, 1., 10.]
    plt.figure(figsize=(18, 7))

    for v in vs:
        k.variance = v
        C = k.K(X, np.array([[0.]]))
        plt.plot(X, C)

    plt.xlabel("x"), plt.ylabel("$\kappa$")
    plt.title("Effects of different variances on the " + k.name)
    plt.legend(labels=vs)
    plt.show()


def plot_lengthscale(k):
    # Our sample space : 100 samples in the interval [-4,4]
    X = np.linspace(-4., 4., 1000)[:, None]  # we use more samples to get a smoother plot at low lengthscales
    plt.figure(figsize=(18, 7))

    ls = [0.25, 0.5, 1., 2., 4.]

    for l in ls:
        k.lengthscale = l
        C = k.K(X, np.array([[0.]]))
        plt.plot(X, C)

    plt.xlabel("x"), plt.ylabel("$\kappa(x,0)$")
    plt.title("Effects of different lengthscales on " + k.name)
    plt.legend(labels=ls)
    plt.show()


def main():
    k = kernel_for_string('RBF', 0.1)
    plot_kernel(k)
    plot_variance(k)
    plot_lengthscale(k)


if __name__ == '__main__':
    main()
