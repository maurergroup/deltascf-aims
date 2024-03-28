import math
from typing import Annotated

import numpy as np
import numpy.typing as npt

from deltascf_aims.utils.hints import BetweenInclusive, GreaterThan


def _schmid_pseudo_voigt(
    domain,
    A,
    m: Annotated[float, BetweenInclusive(0, 1)],
    E,
    omega: Annotated[float, GreaterThan(0)],
    asymmetry=False,
    a=0.2,
    b=0,
) -> npt.NDArray[np.float64]:
    """
    Apply broadening scheme for XPS spectra.

    Reference for the broadening scheme can be found in this paper
    https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/sia.5521

    Parameters
    ----------
    domain : np.array
        Numpy linspace of x range and bin width
    A : float
        Intensity
    m : Annotated[float, BetweenInclusive(0, 1)]
        Gaussian-Lorentzian mixing parameter
    E : float
        Line centre (aka Dirac peak)
    omega : Annotated[float, GreaterThan(0)]
        Full width at half maximum
    asymmetry : bool, optional
        Whether to asymmetrically broaden the spectra
    a : float, optional
        Asymmetry parameter
    b : float, optional
        Asymmetry translation parameter

    Returns
    -------
    V : npt.NDArray[np.float64]
        Broadened spectrum point
    """

    if asymmetry is False:
        V = A * (1 - m) * np.sqrt((4 * np.log(2)) / (math.pi * omega**2)) * np.exp(
            -(4 * np.log(2) / omega**2) * (domain - E) ** 2
        ) + A * m * (1 / (2 * np.pi)) * (omega / ((omega / 2) ** 2 + (domain - E) ** 2))

    else:
        omega_as = 2 * omega / (1 + np.exp(-a * domain - b))

        V = A * (1 - m) * np.sqrt(
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

    return V


def broaden(
    start, stop, A, m, dirac_peaks, omega, asymmetry, a, b
) -> npt.NDArray[np.float64]:
    """Braoden the Dirac delta peaks.

    Parameters
    ----------
    start : int
        Beginning of x-range
    stop : End of x-range
        End of x-range
    A : float
        Intensity
    m : float
        Gaussian-Lorentzian mising
    dirac_peaks : np.array
        Array of Dirac peaks
    omega : float
        Full width at half maximum
    asymmetry : bool
        Whether to asymmetrically broaden the spectra
    a : float
        Asymmetry parameter
    b : float
        Asymmetry translation parameter

    Returns
    -------
    data : npt.NDArray[np.float64]
        Broadened spectrum
    """

    domain = np.linspace(start, stop, 100000)
    data = np.zeros([len(domain)])
    for i in dirac_peaks:
        V = _schmid_pseudo_voigt(domain, A, m, i, omega, asymmetry, a, b)
        data += V

    return data
