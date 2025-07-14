import os
from typing import List, Tuple


def read_ground_energy(calc_path) -> float:
    """
    Get the ground state energy.

    Parameters
    ----------
        calc_path : str
            path to the calculation directory

    Returns
    -------
        grenrgys : float
            ground state energy
    """
    with open(f"{calc_path}ground/aims.out", encoding="utf-8") as ground:
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


def _contains_number(string) -> bool:
    """
    Check if a number is in a string.

    Parameters
    ----------
        string : str
            string to check

    Returns
    -------
        found_string : bool
            True if a number is found, False otherwise
    """
    found_string = False

    for character in string:
        if character.isdigit():
            found_string = True

    return found_string


def read_excited_energy(calc_path, element) -> Tuple[List[float], str]:
    """
    Get the excited state energies.

    Parameters
    ----------
        calc_path : str
            path to the calculation directory
        element : str
            atom to get the excited state energies for

    Returns
    -------
        excienrgys : list[float]
            excited state energies
    """
    dir_list = os.listdir(calc_path)
    energy = "s.c.f. calculation      :"
    excienrgys = []

    # Read each core hole dir
    for directory in dir_list:
        if element in directory and _contains_number(directory):
            # Try reading output file from basis, then projector file structure
            if os.path.exists(f"{calc_path}{directory}/aims.out"):
                with open(
                    f"{calc_path}{directory}/aims.out", errors="ignore"
                ) as out:
                    lines = out.readlines()
            elif os.path.exists(f"{calc_path}{directory}/hole/aims.out"):
                with open(
                    f"{calc_path}{directory}/hole/aims.out", errors="ignore"
                ) as out:
                    lines = out.readlines()
            else:
                lines = []

            for line in lines:
                # Get the energy
                if energy in line:
                    excienrgys.append(float(line.split()[-2]))

    print("Core hole calculated energies (eV):", *excienrgys, sep="\n")

    return excienrgys, element


def calc_delta_scf(element, grenrgys, excienrgys) -> List[float]:
    """
    Calculate delta scf BEs and write to a file.

    Parameters
    ----------
        element : str
            atom to get the excited state energies for
        grenrgys : float
            ground state energy
        excienrgys : List[float]
            excited state energies
    """
    xps = []

    for i in excienrgys:
        xps.append(i - grenrgys)

    print("\nDelta-SCF energies (eV):")

    for i, be in enumerate(xps):
        xps[i] = str(round(be, 3))
        print(xps[i])

    with open(element + "_xps_peaks.txt", "w") as file:
        file.writelines(xps)

    return [float(be) for be in xps]
