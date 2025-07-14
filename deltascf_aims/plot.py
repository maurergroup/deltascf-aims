import glob
import math
import re
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


class XPSSpectrum:
    """
    Parse, calculate, and plot the XPS spectrum.

    ...

    Attributes
    ----------
        run_loc : str
            The location of the FHI-aims run
        targ_at : str
            Atom to calculate the XPS spectrum for
        xy_axis_aims : numpy.ndarray
            Values of the x- and y-axis of the XPS spectrum
        aims_be : float
            Mean average binding energy of the XPS spectrum
        aims_be_line : list
            Vertical line to plot the mean average binding energy
        plot_x : list
            Values of the x-axis of the XPS spectrum to plot
        plot_y : list
            Values of the y-axis of the XPS spectrum to plot
        x_max : int
            Maximum value of the x-axis of the XPS spectrum
        x_min : int
            Minimum value of the x-axis of the XPS spectrum
        y_max : int
            Maximum value of the y-axis of the XPS spectrum

    Methods
    -------
        _find_k_edge_mabe(output=True)
            Find the mean average binding energy of the XPS spectrum
        _get_spectrum_range(gmp)
            Calculate the range of the XPS spectrum to plot
        get_molecule_type(at_spec)
            Get the molecule type from the geometry.in file
        plot(xps, gmp)
            Plot the XPS spectrum and save as pdf and png files.
    """

    def __init__(self, gmp, run_loc, targ_at, print_name=True) -> None:
        """
        Parameters
        ----------
        gmp : float
            Global minimum percentage.
        run_loc : str
            The location of the FHI-aims run.
        targ_at : str
            Atom to calculate the XPS spectrum for.
        print_name : bool
            Whether to seach for the molecule name in the calculation input files.
        """
        self.gmp = gmp
        self.run_loc = run_loc
        self.targ_at = targ_at
        self.print_name = print_name

        # Parse spectrum
        self.xy_axis_aims = np.loadtxt(
            f"{run_loc}/{targ_at}_xps_spectrum.txt", usecols=(0, 1)
        ).T

    def _find_k_edge_mabe(self, output=True) -> None:
        """
        Find the mean average binding energy of the XPS spectrum

        Parameters
        ----------
        output : bool
            Whether to print the mean average binding energy.
        """
        aims_y_max_arg = self.xy_axis_aims[1].argmax()
        self.aims_be = round(self.xy_axis_aims[0][aims_y_max_arg], 4)

        if output:
            print("\nFHI-aims mean average binding energy (MABE):", self.aims_be, "eV")

    def _get_spectrum_range(self) -> None:
        """
        Calculate the range of the XPS spectrum to plot
        """
        glob_max_y = max(self.xy_axis_aims[1])
        glob_min_y = glob_max_y * self.gmp

        # Populate arrays with values above the global minimum
        plot_x = np.where(self.xy_axis_aims[1] > glob_min_y, self.xy_axis_aims[0], 0)
        plot_y = np.where(self.xy_axis_aims[1] > glob_min_y, self.xy_axis_aims[1], 0)

        # Remove 0 values
        self.plot_x = plot_x[plot_x != 0]
        self.plot_y = plot_y[plot_y != 0]

        # Calculate the min and max ranges
        self.x_max = math.floor(max(self.plot_x))
        self.x_min = math.ceil(min(self.plot_x))
        self.y_max = max(self.plot_y)

    def get_molecule_type(self) -> Union[str, None]:
        """
        Get the molecule type from the geometry.in file

        Returns
        -------
        molecule : str
            Molecule type.
        """
        # Get first directory with target atom followed by any number
        # Enable for both basis and projector file structures
        try:
            dirs = " ".join(glob.glob(f"{self.run_loc}/{self.targ_at}*/geometry.in"))
            match = re.findall(rf"{self.targ_at}\d+", dirs)[0]

        except IndexError:
            dirs = " ".join(
                glob.glob(f"{self.run_loc}/{self.targ_at}*/hole/geometry.in")
            )
            match = re.findall(rf"{self.targ_at}\d+", dirs)[0]

        # Enable for both basis and projector file structures
        try:
            with open(f"{self.run_loc}/{match}/geometry.in") as hole_geom:
                lines = hole_geom.readlines()

        except FileNotFoundError:
            with open(f"{self.run_loc}/{match}/hole/geometry.in") as hole_geom:
                lines = hole_geom.readlines()

        try:
            molecule = lines[4].split()[-1]
        except IndexError:
            # If, for example, a custom geometry was used and no molecular
            # specification was made
            molecule = None

        return molecule

    def plot(self, xps, exclude_mabe=False) -> None:
        """
        Plot the XPS spectrum and save as pdf and png files.

        Parameters
        ----------
        xps : list
            List of individual binding energies.
        exclude_mabe : bool
            Whether to exclude the mean average binding energy from the plot.
        """
        # Plot the individual binding energies
        # Include the peak in the legend if first call
        plt.axvline(
            x=xps[0],
            c="#9467bd",  # Purpley colour
            ymax=0.25,
            label="Individual binding energies",
        )

        for peak in xps[1:]:
            plt.axvline(x=peak, c="#9467bd", ymax=0.25)  # Purpley colour

        self._get_spectrum_range()

        # Set general plot parameters
        plt.xlabel("Energy / eV")
        plt.ylabel("Intensity")
        ylims = plt.ylim((0, self.y_max * 1.4))
        # Reverse to match experimental XPS conventions
        plt.xlim(self.x_max, self.x_min)

        # Plot the spectrum
        plt.plot(self.plot_x, self.plot_y, label="Simulated XPS spectrum")

        # Add the mean average binding energy to the plot
        if not exclude_mabe:
            self._find_k_edge_mabe()

            plt.axvline(
                self.aims_be,
                c="grey",
                ymax=self.y_max / ylims[1],
                linestyle="--",
                label=f"MABE = {self.aims_be} eV",
            )

        plt.legend(loc="upper right")

        # Add molecule name to title if given
        if self.print_name:
            molecule = self.get_molecule_type()
            if molecule is not None:
                plt.title(f"XPS spectrum of {molecule}")

        # Save as both a pdf and png
        plt.savefig(f"{self.run_loc}/xps_spectrum.pdf", dpi=300)
        plt.savefig(f"{self.run_loc}/xps_spectrum.png", dpi=300)

    # TODO finish and check this function
    def plot_2(self, xps):
        """
        Plot the invidual binding energies. Include the peak in the legend if first call.
        """
        raise NotImplementedError

        plt.axvline(
            x=xps[0],
            c="#9467bd",  # Purpley colour
            ymax=0.25,
            label="Individual binding energies",
        )

        for peak in xps[1:]:
            plt.axvline(x=peak, c="#9467bd", ymax=0.25)  # Purpley colour

        self._get_spectrum_range()

        # Plot the spectrum
        plt.plot(self.plot_x, self.plot_y, label="Simulated XPS spectrum")

        # Set general plot parameters
        plt.xlabel("Energy / eV")
        plt.ylabel("Intensity")
        plt.ylim((0, self.y_max + 1))
        plt.xlim(
            self.x_max, self.x_min
        )  # Reverse to match experimental XPS conventions
        # plt.xticks(np.arange(self.x_min, self.x_max, 1))
        plt.legend()  # (loc="upper right")

        return plt
