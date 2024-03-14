import glob
import math
import re

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
        x_axis_aims : list
            Values of the x-axis of the XPS spectrum
        y_axis_aims : list
            Values of the y-axis of the XPS spectrum
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

    def __init__(self, gmp, run_loc, targ_at) -> None:
        """
        Parameters
        ----------
            gmp : float
                global minimum percentage
            run_loc : str
                the location of the FHI-aims run
            targ_at : str
                atom to calculate the XPS spectrum for
        """

        self.gmp = gmp
        self.run_loc = run_loc
        self.targ_at = targ_at

        # Parse spectrum
        self.x_axis_aims = np.loadtxt(
            f"{run_loc}/{targ_at}_xps_spectrum.txt", usecols=(0)
        )

        self.y_axis_aims = np.loadtxt(
            f"{run_loc}/{targ_at}_xps_spectrum.txt", usecols=(1)
        )

    def _find_k_edge_mabe(self, output=True) -> None:
        """
        Find the mean average binding energy of the XPS spectrum

        Parameters
        ----------
            output : bool
                whether to print the mean average binding energy
        """

        aims_y_max = self.y_axis_aims.max()
        aims_y_max_arg = self.y_axis_aims.argmax()
        self.aims_be = round(self.x_axis_aims[aims_y_max_arg], 4)
        self.aims_be_line = [
            i for i in np.linspace(-0.6, aims_y_max, num=len(self.y_axis_aims))
        ]

        if output:
            print("\nFHI-aims mean average binding energy (MABE):", self.aims_be, "eV")

    def _get_spectrum_range(self) -> None:
        """
        Calculate the range of the XPS spectrum to plot
        """

        tmp_plot_x = np.array([])
        tmp_plot_y = np.array([])

        glob_max_y = max(self.y_axis_aims)
        glob_min_y = glob_max_y * self.gmp

        for c, y in enumerate(self.y_axis_aims):
            if y > glob_min_y:
                self.plot_x = np.append(tmp_plot_x, self.x_axis_aims[c])
                self.plot_y = np.append(tmp_plot_y, y)

        # Calculate the min and max ranges
        self.x_max = math.floor(max(self.plot_x))
        self.x_min = math.ceil(min(self.plot_x))
        self.y_max = max(self.plot_y)

    def get_molecule_type(self) -> str:
        """
        Get the molecule type from the geometry.in file

        Returns
        -------
            molecule : str
                molecule type
        """

        # Get first directory with target atom followed by any number
        dirs = " ".join(glob.glob(f"{self.run_loc}/{self.targ_at}*/geometry.in"))
        match = re.findall(rf"{self.targ_at}\d+", dirs)[0]

        # Enable for both basis and projector file structures
        try:
            with open(f"{self.run_loc}/{match}/geometry.in", "r") as hole_geom:
                lines = hole_geom.readlines()

        except FileNotFoundError:
            with open(f"{self.run_loc}/{match}/hole/geometry.in", "r") as hole_geom:
                lines = hole_geom.readlines()

        try:
            molecule = lines[4].split()[-1]
        except IndexError:
            # If, for example, a custom geometry was used and no molecular
            # specification was made
            molecule = "custom geometry"

        return molecule

    def plot(self, xps) -> None:
        """
        Plot the XPS spectrum and save as pdf and png files.

        Parameters
        ----------
            xps : list
                list of individual binding energies
            gmp : float
                global minimum percentage
        """

        self._find_k_edge_mabe()

        # Add the mean average binding energy to the plot
        plt.plot(
            np.full((len(self.aims_be_line)), self.aims_be),
            self.aims_be_line,
            c="grey",
            linestyle="--",
            label=f"MABE = {self.aims_be} eV",
        )

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

        # Plot the spectrum
        plt.plot(self.plot_x, self.plot_y, label="Simulated XPS spectrum")
        plt.xlabel("Energy / eV")
        plt.ylabel("Intensity")
        plt.ylim((0, self.y_max + 1))
        plt.xlim(
            self.x_max, self.x_min
        )  # Reverse to match experimental XPS conventions
        plt.xticks(np.arange(self.x_min, self.x_max, 1))
        plt.legend(loc="upper right")

        molecule = self.get_molecule_type()
        if molecule != "None":
            plt.title(f"XPS spectrum of {molecule}")

        # Save as both a pdf and png
        plt.savefig(f"{self.run_loc}/xps_spectrum.pdf")
        plt.savefig(f"{self.run_loc}/xps_spectrum.png")
