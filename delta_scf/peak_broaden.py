import click
import numpy as np


def _gaussian(x, x_mean, broadening):
    gaussian_val = np.sqrt((4 * np.log(2)) / (np.pi * (broadening**2))) * np.exp(
        -((4 * np.log(2)) / (broadening**2)) * (x - x_mean) ** 2
    )

    return gaussian_val


def _lorentzian(x, x_mean, broadening):
    lorentzian_val = (
        (1 / (2 * np.pi)) * (broadening) / (((broadening / 2) ** 2) + (x - x_mean) ** 2)
    )

    return lorentzian_val


def _pseudo_voigt(x, x_mean, broadening, mixing):
    """
    Combines gaussian and lorentzian schemes together
    """

    return (1 - mixing) * _gaussian(x, x_mean, broadening) + mixing * _lorentzian(
        x, x_mean, broadening
    )


def dos_binning(
    eigenvalues,
    broadening=0.75,
    bin_width=0.01,
    mix1=0.0,
    mix2=0.0,
    coeffs=None,
    start=0.0,
    stop=10.0,
    broadening2=None,
    ewid1=10.0,
    ewid2=20.0,
):
    """
    performs binning for a given set of eigenvalues and
    optionally weight coeffs.
    """

    if broadening2 is None:
        broadening2 = broadening
    if coeffs is None:
        coeffs = np.ones(len(eigenvalues))

    lowest_e = start
    highest_e = stop
    num_bins = int((highest_e - lowest_e) / bin_width)
    x_axis = np.zeros([num_bins])
    data = np.zeros([num_bins])

    # Set up x-axis
    for i in range(num_bins):
        x_axis[i] = lowest_e + i * bin_width

    # Get DOS
    sigma = np.zeros((len(eigenvalues)))
    mixing = np.zeros((len(eigenvalues)))

    for ei, e in enumerate(eigenvalues):
        if e <= ewid1:
            sigma[ei] = broadening
            mixing[ei] = mix1
        elif e > ewid2:
            sigma[ei] = broadening2
            mixing[ei] = mix2
        else:
            sigma[ei] = broadening + ((broadening2 - broadening) / (ewid2 - ewid1)) * (
                e - ewid1
            )
            mixing[ei] = mix1 + ((mix2 - mix1) / (ewid2 - ewid1)) * (e - ewid1)

    print()
    with click.progressbar(
        range(num_bins), label="applying pseudo-Voigt broadening..."
    ) as bar:
        for i in bar:
            pseudovoigt_vec = np.zeros((len(eigenvalues)))
            pseudovoigt_vec = (
                _pseudo_voigt(x_axis[i], eigenvalues, sigma, mixing) * coeffs
            )
            data[i] = np.sum(pseudovoigt_vec)

    return x_axis, data
