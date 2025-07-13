from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from ase import Atoms
from ase.build import molecule
from ase.data.pubchem import pubchem_atoms_search

if TYPE_CHECKING:
    from deltascf_aims.core import Start


def add_molecule_identifier(
    start: "Start", atom_specifier: list[int], basis: bool = False
) -> None:
    """
    Add a string to the geometry.in to parse when plotting to identify it.

    Parameters
    ----------
    start: Start
        Instance of Start
    atom_specifier : list[int]
        Atom indices as given in geometry.in
    basis : bool, optional
        Whether a basis calculation is being run
    """
    hole = "" if basis else "/hole"

    with open(
        f"{start.run_loc}/{start.constr_atom}{atom_specifier[0]}{hole}/geometry.in",
    ) as hole_geom:
        lines = hole_geom.readlines()

    # Check that the molecule identifier is not already in the file
    for line in lines:
        if start.spec_mol in line:
            return

    lines.insert(4, f"# {start.spec_mol}\n")

    with open(
        f"{start.run_loc}/{start.constr_atom}{atom_specifier[0]}{hole}/geometry.in",
        "w",
    ) as hole_geom:
        hole_geom.writelines(lines)


def build_geometry(geometry: str) -> Atoms:
    """
    Try getting geometry data from various databases to create a geometry.in file.

    Parameters
    ----------
    geometry : str
        Name or formula of the system to be created

    Returns
    -------
    atoms : Atoms
        Atoms object, or list of atoms objects

    Raises
    ------
    SystemExit
        Exit the program if the system is not found in any database
    """
    try:
        atoms = molecule(geometry)
        print("molecule found in ASE database")
    except KeyError:
        print("molecule not found in ASE database, searching PubChem...")
    else:
        return atoms

    try:
        atoms = pubchem_atoms_search(name=geometry)
        if isinstance(atoms, list):
            atoms = atoms[0]
        print("molecule found as a PubChem name")
    except ValueError:
        print(f"{geometry} not found in PubChem name")
    else:
        return atoms

    try:
        atoms = pubchem_atoms_search(cid=geometry)
        if isinstance(atoms, list):
            atoms = atoms[0]
        print("molecule found in PubChem CID")
    except ValueError:
        print(f"{geometry} not found in PubChem CID")
    else:
        return atoms

    try:
        atoms = pubchem_atoms_search(smiles=geometry)
        if isinstance(atoms, list):
            atoms = atoms[0]
        print("molecule found in PubChem SMILES")
    except ValueError as err:
        print(f"{geometry} not found in PubChem smiles")
        print(f"{geometry} not found in PubChem or ASE database")
        print("aborting...")
        raise SystemExit from err
    else:
        return atoms


def get_all_elements() -> list[str]:
    """
    Get a list of all element symbols supported by FHI-aims.

    Returns
    -------
    elements : list[str]
        Element symbols
    """
    # Find the root directory of the package
    current_path = Path(__file__).parent.resolve()

    # Get all supported elements in FHI-aims
    with open(f"{current_path}/elements.yml") as elements_file:
        return yaml.load(elements_file, Loader=yaml.SafeLoader)


def get_atoms(
    geometry: Path,
    constr_atom: str,
) -> list[int]:
    """
    Get the atom indices to constrain from the geometry file.

    Parameters
    ----------
    geometry : pathlib.Path
        Path to the geometry file
    constr_atom : str
        list of elements to constrain

    Returns
    -------
    list[int]
        list of atom indices to constrain

    Raises
    ------
    Click.MissingParameter
        A required parameter has not been given
    ValueError
        An invalid parameter has been given
    """
    elements = get_all_elements()
    atom_specifier = []

    if constr_atom not in elements:
        raise ValueError("invalid element specified")

    print("Calculating all target atoms in geometry.in")

    # Constrain all atoms of the target element
    with geometry.open() as geom_in:
        atom_counter = 0

        for line in geom_in:
            spl = line.split()

            if len(spl) > 0 and "atom" in spl[0]:
                atom_counter += 1
                element = spl[-1]  # Identify atom
                identifier = spl[0]  # Extra check that line is an atom

                if "atom" in identifier and element == constr_atom:
                    atom_specifier.append(atom_counter)

    print("Specified atom indices:", atom_specifier)

    return atom_specifier


def get_element_symbols(geom: Path, spec_at_constr: list[int]) -> list[str]:
    """
    Find the element symbols from specified atom indices in a geometry file.

    Parameters
    ----------
    geom : Path
        Path to the geometry file
    spec_at_constr : list[int]
        list of atom indices

    Returns
    -------
    list[str]
        list of element symbols
    """
    with geom.open() as geom_file:
        lines = geom_file.readlines()

    atom_lines = []

    # Copy only the lines which specify atom coors into a new list
    for line in lines:
        spl = line.split()
        if len(line) > 0 and spl[0] == "atom":
            atom_lines.append(line)

    element_symbols = []

    # Get the element symbols from the atom coors
    # Uniquely add each element symbol
    for atom in spec_at_constr:
        element = atom_lines[atom].split()[-1]

        if element not in element_symbols:
            element_symbols.append(element)

    return element_symbols
