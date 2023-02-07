"""Utilities which are used in aims_dscf"""

import sys

from ase.build import molecule
from ase.calculators.aims import Aims
from ase.data.pubchem import pubchem_atoms_search
from click import MissingParameter


class MainUtils:
    """Various static methods used in aims_dscf"""

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def check_geom_constraints(geom_file):
        """Check if there are any constrain_relaxation keywords in geometry.in"""

        with open(geom_file, "r") as geom:
            lines = geom.readlines()

        for line in lines:
            if "constrain_relaxation" in line:
                print("'constrain_relaxation' keyword found in geometry.in")
                print("Ensure that no atoms are fixed in the geometry.in file")
                print(
                    "The geometry of the structure should have already been relaxed before any SP calculations"
                )
                print("Aborting...")
                sys.exit(1)

    @staticmethod
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

    @staticmethod
    def print_ks_states(run_loc):
        """Print the KS states for the different spin states"""

        # Parse the output file
        with open(f"{run_loc}/ground/aims.out", "r") as aims:
            lines = aims.readlines()

        su_eigs_start_line = None
        sd_eigs_start_line = None

        for num, content in enumerate(lines):
            if "Spin-up eigenvalues" in content:
                su_eigs_start_line = num
            if "Spin-down eigenvalues" in content:
                sd_eigs_start_line = num

        # Check that KS states were found
        if su_eigs_start_line is None:
            print("No spin-up KS states found")
            print("Did you run a spin polarised calculation?")
            sys.exit(1)

        if sd_eigs_start_line is None:
            print("No spin-down KS states found")
            print("Did you run a spin polarised calculation?")
            sys.exit(1)

        su_eigs = []
        sd_eigs = []

        # Save the KS states into lists
        for num, content in enumerate(lines[su_eigs_start_line + 2 :]):
            spl = content.split()

            if len(spl) != 0:
                su_eigs.append(content)
            else:
                break

        for num, content in enumerate(lines[sd_eigs_start_line + 2 :]):
            spl = content.split()

            if len(spl) != 0:
                sd_eigs.append(content)
            else:
                break

        # Print the KS states
        print("Spin-up KS eigenvalues:\n")
        print(*su_eigs, sep="")

        print("Spin-down KS eigenvalues:\n")
        print(*sd_eigs, sep="")
