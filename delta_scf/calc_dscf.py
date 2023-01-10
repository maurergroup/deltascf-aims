import os


def read_ground(calc_path):
    """Get the ground state energy."""
    with open(f"{calc_path}ground/aims.out", "r", encoding="utf-8") as ground:
        lines = ground.readlines()

    grenrgys = None

    for line in lines:
        # Get the energy
        if "s.c.f. calculation      :" in line:
            grenrgys = float(line.split()[-2])

    if grenrgys == None:
        raise ValueError("No ground state energy found.")

    print("Ground state calculated energy (eV):")
    print(round(grenrgys, 3))
    print()

    return grenrgys


def contains_number(string):
    """Check if a number is in a string."""
    for character in string:
        if character.isdigit():
            return True


def read_atoms(calc_path, element, contains_number):
    """Get the excited state energies."""
    dir_list = os.listdir(calc_path)
    energy = "s.c.f. calculation      :"
    excienrgys = []

    # Read each core hole dir
    atom_counter = 0
    for directory in dir_list:
        if element in directory and contains_number(directory) is True:
            atom_counter += 1

            with open(
                calc_path + directory + "/hole/aims.out", "r", encoding="utf-8"
            ) as out:
                lines = out.readlines()

            for line in lines:
                # Get the energy
                if energy in line:
                    excienrgys.append(float(line.split()[-2]))

    # Remove duplicate binding energies from list
    excienrgys = list(dict.fromkeys(excienrgys))

    print("Core hole calculated energies (eV):", *excienrgys, sep="\n")

    return element, excienrgys


def calc_delta_scf(element, grenrgys, excienrgys):
    """Calculate delta scf and write to a file."""
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