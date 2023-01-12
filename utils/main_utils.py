"""Utilities which are used in aims_dscf"""

import sys

from ase.build import molecule
from ase.calculators.aims import Aims
from ase.constraints import FixAtoms
from ase.data.pubchem import pubchem_atoms_search
from click import MissingParameter

from directional_constraint import DirectionalConstraint as dc


def build_geometry(geometry):
    """Check different databases to create a geometry.in"""

    try:
        atoms = molecule(geometry)
        print("molecule found in ASE database")
        return atoms
    except KeyError:
        print("molecule not found in ASE database, searching PubChem...")

    try:
        atoms = pubchem_atoms_search(name=geometry)
        print("molecule found as a PubChem name")
        return atoms
    except ValueError:
        print(f"{geometry} not found in PubChem name")

    try:
        atoms = pubchem_atoms_search(cid=geometry)
        print("molecule found in PubChem CID")
        return atoms
    except ValueError:
        print(f"{geometry} not found in PubChem CID")

    try:
        atoms = pubchem_atoms_search(smiles=geometry)
        print("molecule found in PubChem SMILES")
        return atoms
    except ValueError:
        print(f"{geometry} not found in PubChem smiles")
        print(f"{geometry} not found in PubChem or ASE database")
        print("aborting...")
        sys.exit(1)


def check_geom_constraints(geom_file):
    """Check if there are any constrain_relaxation keywords in geometry.in"""

    with open(geom_file, "r") as geom:
        lines = geom.readlines()

    for line in lines:
        for coor in ["x", "y", "z"]:
            if f"constrain_relaxation {coor}" in line:
                # TODO
                pass


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
