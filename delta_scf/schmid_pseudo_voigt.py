import math

import numpy as np


def _schmid_pseudo_voigt(domain, A, m, E, omega, asymmetry, a, b):
    """
    Apply broadening scheme for XPS spectra
    https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/sia.5521

    domain = linspace of x range and bin width
    A = intensity
    m = Gaussian-Lorentzian mixing parameter
    E = line centre (aka dirac peak)
    omega = full width at half maximum (omega > 0)
    asymmetry = True or False
    a = asymmetry parameter
    b = asymmetry translation parameter
    """

    if asymmetry is False:
        return A * (1 - m) * np.sqrt((4 * np.log(2)) / (math.pi * omega**2)) * np.exp(
            -(4 * np.log(2) / omega**2) * (domain - E) ** 2
        ) + A * m * (1 / (2 * np.pi)) * (omega / ((omega / 2) ** 2 + (domain - E) ** 2))

    else:
        omega_as = 2 * omega / (1 + np.exp(-a * domain - b))

        return A * (1 - m) * np.sqrt(
            (4 * np.log(2))
            / (np.pi * ((2 * omega_as) / (1 + np.exp(-a * ((domain - E) - b)))) ** 2)
        ) * np.exp(
            -(
                4
                * np.log(2)
                / (2 * omega_as / (1 + np.exp(-a * ((domain - E) - b)))) ** 2
            )
            * (domain - E) ** 2
        ) + A * m * (
            1 / (2 * np.pi)
        ) * (
            (2 * omega_as / (1 + np.exp(-a * ((domain - E) - b))))
            / (
                (((2 * omega_as) / (1 + np.exp(-a * ((domain - E) - b)))) / 2) ** 2
                + (domain - E) ** 2
            )
        )


def broaden(start, stop, A, m, dirac_peaks, omega, asymmetry, a, b):
    """
    Broaden dirac delta peaks

    start = beginning of x range
    stop = end of x range
    dirac_peaks = list of dirac delta peaks
    eta = full width at half maximum (eta > 0)
    """
    domain = np.linspace(start, stop, 100000)
    data = np.zeros([len(domain)])
    for i in dirac_peaks:
        V = _schmid_pseudo_voigt(domain, A, m, i, omega, asymmetry, a, b)
        data += V

    return data, domain
