import sys

from ase.build import molecule
from ase.calculators.aims import Aims
from ase.data.pubchem import pubchem_atoms_search
from click import MissingParameter


def build_geometry(spec_mol):
    """Check different databases to create a geometry.in"""

    try:
        atoms = molecule(spec_mol)
        print("molecule found in ASE database")
        return atoms
    except KeyError:
        print("molecule not found in ASE database, searching PubChem...")

    try:
        atoms = pubchem_atoms_search(name=spec_mol)
        print("molecule found as a PubChem name")
        return atoms
    except ValueError:
        print(f"{spec_mol} not found in PubChem name")

    try:
        atoms = pubchem_atoms_search(cid=spec_mol)
        print("molecule found in PubChem CID")
        return atoms
    except ValueError:
        print(f"{spec_mol} not found in PubChem CID")

    try:
        atoms = pubchem_atoms_search(smiles=spec_mol)
        print("molecule found in PubChem SMILES")
        return atoms
    except ValueError:
        print(f"{spec_mol} not found in PubChem smiles")
        print(f"{spec_mol} not found in PubChem or ASE database")
        print("aborting...")
        sys.exit(1)


def create_calc(procs, binary, species):
    """Create an ASE calculator object"""

    aims_calc = Aims(
        xc="pbe",
        spin="collinear",
        default_initial_moment=0,
        aims_command=f"mpirun -n {procs} {binary}",
        species_dir=f"{species}defaults_2020/tight/",
    )

    return aims_calc


def check_args(*args):
    """Check if required arguments are specified"""
    def_args = locals()

    for arg in def_args["args"]:
        if arg[1] is None:
            if arg[0] == "spec_mol":
                # Convert to list and back to assign to tuple
                arg = list(arg)
                arg[0] = "molecule"
                arg = tuple(arg)

            raise MissingParameter(param_hint=f"'--{arg[0]}'", param_type="option")
