from pathlib import Path


def read_ground_energy(calc_path: Path) -> float:
    """
    Get the ground state energy.

    Parameters
    ----------
    calc_path : pathlib.Path
        path to the calculation directory

    Returns
    -------
    float
        ground state energy
    """
    with (calc_path / "ground" / "aims.out").open() as ground:
        lines = ground.readlines()

    grenrgys = None

    for line in lines:
        # Get the energy
        if "s.c.f. calculation      :" in line:
            grenrgys = float(line.split()[-2])

    if grenrgys is None:
        raise ValueError("No ground state energy found.")

    print("Ground state calculated energy (eV):")
    print(round(grenrgys, 3))
    print()

    return grenrgys


def _contains_number(string: str) -> bool:
    """
    Check if a number is in a string.

    Parameters
    ----------
    string : str
        String to check

    Returns
    -------
    bool
        Whether the string contains a number
    """
    found_string = False

    for character in string:
        if character.isdigit():
            found_string = True

    return found_string


def read_excited_energy(calc_path: Path, element: str) -> tuple[list[float], str]:
    """
    Get the excited state energies.

    Parameters
    ----------
    calc_path : pathlib.Path
        Path to the calculation directory
    element : str
        Atom to get the excited state energies for

    Returns
    -------
    tuple[list[float], str]
        excited state energies
    """
    dir_list = [d for d in calc_path.iterdir() if d.is_dir()]
    energy = "s.c.f. calculation      :"
    excienrgys = []

    # Read each core hole dir
    for directory in dir_list:
        if element in directory.name and _contains_number(directory.name):
            # Try reading output file from basis, then projector file structure
            try:
                with directory.joinpath("aims.out").open() as out:
                    lines = out.readlines()
            except FileNotFoundError:
                try:
                    with directory.joinpath("hole", "aims.out").open() as out:
                        lines = out.readlines()
                except FileNotFoundError:
                    lines = []

            excienrgys.extend(
                [float(line.split()[-2]) for line in lines if energy in line]
            )

    print("Core hole calculated energies (eV):", *excienrgys, sep="\n")

    return excienrgys, element


def calc_delta_scf(
    element: str, grenrgys: float, excienrgys: list[float]
) -> list[float]:
    """
    Calculate delta scf BEs and write to a file.

    Parameters
    ----------
    element : str
        Atom to get the excited state energies for
    grenrgys : float
        Ground state energy
    excienrgys : list[float]
        Excited state energies

    Returns
    -------
    list[float]
        Delta-SCF binding energies
    """
    xps = []

    xps.extend([i - grenrgys for i in excienrgys])

    print("\nDelta-SCF energies (eV):")

    for i, be in enumerate(xps):
        xps[i] = str(round(be, 3))
        print(xps[i])

    with open(element + "_xps_peaks.txt", "w") as file:
        file.writelines(xps)

    return [float(be) for be in xps]
