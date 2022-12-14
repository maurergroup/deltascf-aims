import math

import matplotlib.pyplot as plt
import numpy as np


def sim_xps_spectrum(targ_at):
    x_axis_aims = np.loadtxt(f"test_dirs/{targ_at}_xps_spectrum.txt", usecols=(0))
    y_axis_aims = np.loadtxt(f"test_dirs/{targ_at}_xps_spectrum.txt", usecols=(1))

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

    for c, y in enumerate(y_axis_aims):
        if y > 0.01:
            plot_x.append(x_axis_aims[c])
            plot_y.append(y)

    # Calculate the min and max ranges
    x_max = math.floor(max(plot_x))
    x_min = math.ceil(min(plot_x))
    y_max = max(plot_y)

    # Get the type of molecule
    with open(f"test_dirs/{targ_at}1/hole/geometry.in", "r") as hole_geom:
        lines = hole_geom.readlines()

    molecule = lines[4].split()[-1]

    # Plot the spectrum
    plt.plot(plot_x, plot_y, label="Simulated XPS spectrum")
    plt.ylim((0, y_max + 1))
    plt.xlim(x_max, x_min)  # Reverse to match experimental XPS conventions
    plt.xticks(np.arange(x_min, x_max, 1))
    plt.legend(loc="upper right")
    plt.title(f"XPS spectrum of {molecule}")

    plt.savefig(f"test_dirs/xps_spectrum.pdf")
    plt.savefig(f"test_dirs/xps_spectrum.png")
    plt.show()
