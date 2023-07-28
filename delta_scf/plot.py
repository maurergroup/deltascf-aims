import math

import matplotlib.pyplot as plt
import numpy as np


class Plot:
    """Plot the XPS spectrum"""

    @staticmethod
    def sim_xps_spectrum(xps, run_loc, targ_at, at_spec, gmp):
        x_axis_aims = np.loadtxt(f"{run_loc}/{targ_at}_xps_spectrum.txt", usecols=(0))
        y_axis_aims = np.loadtxt(f"{run_loc}/{targ_at}_xps_spectrum.txt", usecols=(1))

        # Find the k-edge MABE
        aims_y_max = y_axis_aims.max()
        aims_y_max_arg = y_axis_aims.argmax()
        aims_be = round(x_axis_aims[aims_y_max_arg], 4)
        aims_be_line = [i for i in np.linspace(-0.6, aims_y_max, num=len(y_axis_aims))]

        print("\nFHI-aims mean average binding energy (MABE):", aims_be, "eV")

        # Plot everything
        plt.xlabel("Energy / eV")
        plt.ylabel("Intensity")

        plt.plot(
            np.full((len(aims_be_line)), aims_be),
            aims_be_line,
            c="grey",
            linestyle="--",
            label=f"MABE = {aims_be} eV",
        )

        # Find the range of the spectrum to plot
        plot_x = []
        plot_y = []

        glob_max_y = max(y_axis_aims)
        glob_min_y = glob_max_y * gmp

        for c, y in enumerate(y_axis_aims):
            if y > glob_min_y:
                plot_x.append(x_axis_aims[c])
                plot_y.append(y)

        # Calculate the min and max ranges
        x_max = math.floor(max(plot_x))
        x_min = math.ceil(min(plot_x))
        y_max = max(plot_y)

        # Get the type of molecule
        # Enable for both basis and projector file structures
        try:
            with open(f"{run_loc}/{targ_at}{at_spec}/geometry.in", "r") as hole_geom:
                lines = hole_geom.readlines()

        except FileNotFoundError:
            with open(
                f"{run_loc}/{targ_at}{at_spec}/hole/geometry.in", "r"
            ) as hole_geom:
                lines = hole_geom.readlines()

        molecule = lines[4].split()[-1]

        # Plot the individual binding energies
        first_dirac = True
        for peak in xps:
            # Include the peak in the legend if first call
            if first_dirac is True:
                plt.axvline(
                    x=peak,
                    c="#9467bd",
                    ymax=0.25,
                    label="Individual binding energies",
                )
                first_dirac = False
            else:
                plt.axvline(x=peak, c="#9467bd", ymax=0.25)

        # Plot the spectrum
        plt.plot(plot_x, plot_y, label="Simulated XPS spectrum")
        plt.ylim((0, y_max + 1))
        plt.xlim(x_max, x_min)  # Reverse to match experimental XPS conventions
        plt.xticks(np.arange(x_min, x_max, 1))
        plt.legend(loc="upper right")

        if molecule != "None":
            plt.title(f"XPS spectrum of {molecule}")

        plt.savefig(f"{run_loc}/xps_spectrum.pdf")
        plt.savefig(f"{run_loc}/xps_spectrum.png")
        plt.show()
