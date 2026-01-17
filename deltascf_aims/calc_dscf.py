from pathlib import Path


def parse_energy(aims_out: Path) -> float:
    """
    Get the final energy from an FHI-aims DFT calculation.

    Parameters
    ----------
    aims_out : Path
       Path to DFT aims output file

    Returns
    -------
    float
        Total DFT energy

    Raises
    ------
    ValueError
        If the energy could not be found in the output file
    """
    # 1000 lines * ~152 bytes/line = 160000
    # Set to 160000 to be safe
    tail_size = 160000
    target = "| Total energy of the DFT / Hartree-Fock s.c.f. calculation"
    energy = None

    with aims_out.open("rb") as f:
        try:
            f.seek(-tail_size, 2)  # Move to the last tail_size bytes
        except OSError:
            f.seek(0)  # File is smaller than tail_size, go to the start

        # Decode and ignore errors at cut point
        lines = f.read().decode("utf-8", errors="ignore")

    # Iterate in reverse to find the last occurrence of the energy
    for line in reversed(lines.splitlines()):
        if target in line:
            try:
                energy = float(line.split()[-2])
            except (ValueError, IndexError):
                continue
            else:
                return energy

    if energy is None:
        raise ValueError("Could not find total energy in the output file.")

    return energy


def calc_delta_scf(
    element: str, ground: float | list[float], excited: list[float]
) -> list[float]:
    """
    Calculate Delta-SCF binding energies.

    Parameters
    ----------
    element : str
        Atom to get the excited state energies for
    ground : float | list[float]
        Ground state energy(s)
    excited : list[float]
        Excited state energies

    Returns
    -------
    list[float]
        Delta-SCF binding energies
    """
    if isinstance(ground, list):
        xps = [round(i - j, 2) for i, j in zip(excited, ground, strict=True)]

    else:
        xps = [round(i - ground, 2) for i in excited]

    print("\nDelta-SCF energies (eV):")
    print(*xps, sep="\n")

    with Path(f"{element}_xps_peaks.txt").open("w") as file:
        file.writelines(f"{el}\n" for el in xps)

    return xps
