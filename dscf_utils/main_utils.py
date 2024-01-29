"""Utilities which are used in aims_dscf"""

import glob
import os
import warnings
from sys import platform
from typing import List, Union

import numpy as np
import yaml
from ase import Atoms
from ase.build import molecule
from ase.calculators.aims import Aims
from ase.data.pubchem import pubchem_atoms_search
from ase.io import write
from click import MissingParameter

from delta_scf.force_occupation import ForceOccupation as fo


def build_geometry(geometry) -> Union[Atoms, List[Atoms], None]:
    """
    Try various databases to create a geometry.in file.

    Parameters
    ----------
        geometry : str
            Name or formula of the system to be created

    Returns
    -------
        atoms : Union[Atoms, List[Atoms], None]
            Atoms object, list of atoms objects, or None if the system is not found in
            any database
    """

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
        raise SystemExit


def check_args(*args) -> None:
    """
    Check if required arguments are specified.

    Parameters
    ----------
        args : tuple
            arbitrary number of arguments to check
    """

    # TODO: Check if this works with the locals() commented out below
    # def_args = locals()
    # for arg in def_args["args"]:
    for arg in args:
        if arg[1] is None:
            if arg[0] == "spec_mol":
                # Convert to list and back to assign to tuple
                arg = list(arg)
                arg[0] = "molecule"
                arg = tuple(arg)

            raise MissingParameter(param_hint=f"'--{arg[0]}'", param_type="option")


def check_geom(geom_file) -> bool:
    """
    Check for any constrain_relaxation keywords in geometry.in.

    Parameters
    ----------
        geom_file : str
            path to geometry.in file

    Returns
    -------
        lattice_vecs : bool
            True if lattice vectors are found, False otherwise
    """

    lattice_vecs = False

    for line in geom_file:
        if "lattice_vector" in line:
            lattice_vecs = True

        if "constrain_relaxation" in line:
            print("'constrain_relaxation' keyword found in geometry.in")
            print("ensure that no atoms are fixed in the geometry.in file")
            print(
                "the geometry of the structure should have already been relaxed before "
                "any SP calculations"
            )
            print("aborting...")
            raise SystemExit

    return lattice_vecs


def check_control_k_grid(control_file) -> bool:
    """
    Check if there is a k_grid in the control.in.

    Parameters
    ----------
        control_file : str
            path to control.in file

    Returns
    -------
        k_grid : bool
            True if k_grid input parameter is found, False otherwise
    """

    k_grid = False

    for line in control_file:
        if "k_grid" in line:
            k_grid = True

    return k_grid


def convert_opts_to_dict(opts, pbc) -> dict:
    """
    Convert the control options from a tuple to a dictionary.

    Parameters
    ----------
        opts : tuple
            tuple of control options
        pbc : list
            tuple of k-points

    Returns
    -------
        opts_dict : dict
            dictionary of control options
    """

    opts_dict = {}

    for opt in opts:
        spl = opt.split(sep="=")

        opts_dict[spl[0]] = spl[1]

    # Also add k_grid if given
    if pbc is not None:
        opts_dict.update({"k_grid": pbc})

    return opts_dict


def create_calc(procs, binary, species, int_grid) -> Aims:
    """
    Create an ASE calculator object

    Parameters
    ----------
        procs : int
            number of processors to use
        binary : str
            path to aims binary
        species : str
            path to species directory
        int_grid : str
            basis set density

    Returns
    -------
        aims_calc : Aims
            ASE calculator object
    """

    # Choose some sane defaults
    aims_calc = Aims(
        xc="pbe",
        spin="collinear",
        default_initial_moment=0,
        aims_command=f"mpirun -n {procs} {binary}",
        species_dir=f"{species}/defaults_2020/{int_grid}/",
    )

    return aims_calc


def print_ks_states(run_loc) -> None:
    """
    Print the KS states for the different spin states.

    Parameters
    ----------
        run_loc : str
            path to the calculation directory
    """

    # Parse the output file
    with open(f"{run_loc}/ground/aims.out", "r") as aims:
        lines = aims.readlines()

    su_eigs_start_line = None
    sd_eigs_start_line = None

    for num, content in enumerate(lines):
        if "Spin-up eigenvalues" in content:
            su_eigs_start_line = num
            if "K-point:" in lines[num + 1]:
                su_eigs_start_line += 1

        if "Spin-down eigenvalues" in content:
            sd_eigs_start_line = num
            if "K-point:" in lines[num + 1]:
                sd_eigs_start_line += 1

    # Check that KS states were found
    if su_eigs_start_line is None:
        print("No spin-up KS states found")
        print("Did you run a spin polarised calculation?")
        raise SystemExit

    if sd_eigs_start_line is None:
        print("No spin-down KS states found")
        print("Did you run a spin polarised calculation?")
        raise SystemExit

    su_eigs = np.array([])
    sd_eigs = np.array([])

    # Save the KS states into lists
    for num, content in enumerate(lines[su_eigs_start_line + 2 :]):
        spl = content.split()

        if len(spl) != 0:
            np.append(su_eigs, content)
        else:
            break

    for num, content in enumerate(lines[sd_eigs_start_line + 2 :]):
        spl = content.split()

        if len(spl) != 0:
            np.append(sd_eigs, content)
        else:
            break

    # Print the KS states
    print("Spin-up KS eigenvalues:\n")
    print(*su_eigs, sep="")

    print("Spin-down KS eigenvalues:\n")
    print(*sd_eigs, sep="")


def set_env_vars() -> None:
    """
    Set environment variables for running aims.
    """

    os.system("export OMP_NUM_THREADS=1")
    os.system("export MKL_NUM_THREADS=1")
    os.system("export MKL_DYNAMIC=FALSE")

    if platform == "linux" or platform == "linux2":
        os.system("ulimit -s unlimited")
    elif platform == "darwin":
        os.system("ulimit -s hard")
    else:
        warnings.warn("OS not supported, please ensure ulimit is set to unlimited")


def write_control(run_loc, control_opts, atoms, int_grid, defaults) -> None:
    """
    Write a control.in file

    Parameters
    ----------
        run_loc : str
            path to the calculation directory
        control_opts : dict
            dictionary of control options
        atoms : Atoms
            ASE atoms object
        int_grid : str
            basis set density
        defaults : str
            path to species_defaults directory
    """

    # Firstly create the control file
    os.system(f"touch {run_loc}/ground/control.in")

    # Convert any keys given as tuples to strings
    for i in control_opts.items():
        if type(i[1]) == tuple:
            control_opts[i[0]] = " ".join(str(j) for j in i[1])

    # Use the static method from ForceOccupation
    lines = fo.change_control_keywords(f"{run_loc}/ground/control.in", control_opts)

    with open(f"{run_loc}/ground/control.in", "w") as control:
        control.writelines(lines)

    # Then add the basis set
    elements = list(set(atoms.get_chemical_symbols()))

    for el in elements:
        basis_set = glob.glob(f"{defaults}/ch_basis_sets/{int_grid}/*{el}_default")[0]
        os.system(f"cat {basis_set} >> {run_loc}/ground/control.in")


class GroundCalc:
    """
    Setup and run a ground state calculation.

    ...

    Attributes
    ----------
        run_loc : str
            path to the calculation directory
        atoms : Atoms
            ASE atoms object
        basis_set : str
            basis set density
        species : str
            path to species directory
        ase : bool
            whether to use ASE
        hpc : bool
            whether to run on a HPC

    Methods
    -------
        _set_env_vars()
            Set environment variables for running aims.
        _setup_files_and_dirs(geom_inp, control_inp)
            Setup the ground calculation files and directories.
        add_extra_basis_fns(constr_atom)
            Add additional basis functions to the basis set.
        _with_ase(calc, control_opts, l_vecs)
            Run the ground state calculation using ASE.
        _without_ase(print_output, nprocs, binary)
            Run the ground state calculation without ASE.
        run(geom_inp, control_inp, constr_atom, calc, control_opts, l_vecs, print_output, nprocs, binary)
            Run the ground state calculation.
    """

    def __init__(
        self,
        run_loc,
        atoms,
        basis_set,
        species,
        ase,
        hpc,
    ):
        self.run_loc = run_loc
        self.atoms = atoms
        self.basis_set = basis_set
        self.species = species
        self.ase = ase
        self.hpc = hpc

    def _setup_files_and_dirs(self, geom_inp, control_inp) -> None:
        """
        Setup the ground calculation files and directories.

        Parameters
        ----------
            geom_inp : str
                path to the geometry.in file
            control_inp : str
                path to the control.in file
        """

        # Create the ground directory if it doesn't already exist
        os.system(f"mkdir -p {self.run_loc}/ground")

        # Write the geometry file if the system is specified through CLI
        if geom_inp is None and control_inp is not None:
            write(f"{self.run_loc}/geometry.in", self.atoms, format="aims")

        # Copy the geometry.in and control.in files to the ground directory
        if control_inp is not None:
            os.system(f"cp {control_inp} {self.run_loc}/ground")

        if geom_inp is not None:
            os.system(f"cp {geom_inp} {self.run_loc}/ground")

    def add_extra_basis_fns(self, constr_atom):
        """
        Add additional basis functions to the basis set.

        Parameters
        ----------
            constr_atom : str
                element symbol of the constrained atom
        """

        basis_file = glob.glob(
            f"{self.species}/defaults_2020/{self.basis_set}/*{constr_atom}_default"
        )[0]
        current_path = os.path.dirname(os.path.realpath(__file__))

        with open(basis_file, "r") as basis_functions:
            control_content = basis_functions.readlines()

        with open(f"{current_path}/../delta_scf/elements.yml", "r") as elements:
            elements = yaml.load(elements, Loader=yaml.SafeLoader)

        new_content = fo.add_additional_basis(
            current_path, elements, control_content, constr_atom
        )

        # Create a new directory for modified basis sets
        new_dir = f"{self.species}/ch_self.basis_sets/{self.basis_set}/"

        if os.path.exists(new_dir):
            os.system(f"rm {new_dir}/*")
        else:
            os.system(f"mkdir -p {self.species}/ch_self.basis_sets/{self.basis_set}/")

        os.system(
            f"cp {basis_file} {self.species}/ch_self.basis_sets/{self.basis_set}/"
        )
        new_basis_file = glob.glob(
            f"{self.species}/ch_self.basis_sets/{self.basis_set}/*{constr_atom}_default"
        )[0]

        if new_content is not None:
            with open(new_basis_file, "w") as basis_functions:
                basis_functions.writelines(new_content)

        # Copy atoms from the original basis set directory to the new one
        chem_symbols = list(set(self.atoms.get_chemical_symbols()))

        for atom in chem_symbols:
            if atom != constr_atom:
                os.system(
                    f"cp {self.species}/defaults_2020/{self.basis_set}/*{atom}_default "
                    f"{self.species}/ch_self.basis_sets/{self.basis_set}/"
                )

    def _with_ase(self, calc, control_opts, l_vecs) -> None:
        """
        Run the ground state calculation using ASE.

        Parameters
        ----------
            control_opts : dict
                dictionary of control options
            l_vecs : list
                lattice vectors
        """

        # Change the defaults if any are specified by the user
        # Update with all control options from the calculator
        calc.set(**control_opts)
        control_opts = calc.parameters

        # Add the lattice vectors if periodic
        if l_vecs is not None:
            self.atoms.set_pbc(l_vecs)

        if self.hpc:
            # Prevent species dir from being written
            control_opts.pop("species_dir")

            print("writing geometry.in file...")
            write(
                f"{self.run_loc}/ground/geometry.in", images=self.atoms, format="aims"
            )

            print("writing control.in file...")
            write_control(
                self.run_loc, control_opts, self.atoms, self.basis_set, self.species
            )

        else:
            print("running calculation...")
            self.atoms.get_potential_energy()
            # Move files to ground directory
            os.system(f"cp geometry.in control.in {self.run_loc}/ground/")
            os.system(f"mv aims.out parameters.ase {self.run_loc}/ground/")

    def _without_ase(self, print_output, nprocs, binary) -> None:
        """
        Run the ground state calculation without ASE.

        Parameters
        ----------
            print_output : bool
                whether to print the output of the calculation
            nprocs : int
                number of processors to use
            binary : str
                path to the aims binary
        """

        print("running calculation...")

        if print_output:  # Show live output of calculation
            os.system(
                f"cd {self.run_loc}/ground && mpirun -n {nprocs} {binary} | tee aims.out"
            )

        else:
            os.system(
                f"cd {self.run_loc}/ground && mpirun -n {nprocs} {binary} > aims.out"
            )

    def run(
        self,
        geom_inp,
        control_inp,
        constr_atom,
        calc,
        control_opts,
        l_vecs,
        print_output,
        nprocs,
        binary,
    ) -> None:
        """
        Run the ground state calculation.

        Parameters
        ----------
            geom_inp : str
                path to the geometry.in file
            control_inp : str
                path to the control.in file
            constr_atom : str
                element symbol of the constrained atom
            calc : Aims
                ASE calculator object
            control_opts : dict
                dictionary of control options
            l_vecs : list
                lattice vectors
            print_output : bool
                whether to print the output of the calculation
            nprocs : int
                number of processors to use
            binary : str
                path to the aims binary
        """

        # Export environment variables
        set_env_vars()

        # Setup the files and directories
        GroundCalc._setup_files_and_dirs(self, geom_inp, control_inp)

        # Run the ground state calculation
        if os.path.isfile(f"{self.run_loc}/ground/aims.out") is False:
            GroundCalc.add_extra_basis_fns(self, constr_atom)

            # Use ASE
            if self.ase:
                GroundCalc._with_ase(self, calc, control_opts, l_vecs)

            # Don't use ASE
            elif not self.hpc:  # Don't use ASE
                GroundCalc._without_ase(self, print_output, nprocs, binary)

            # Print the KS states from aims.out so it is easier to specify the
            # KS states for the hole calculation
            if not self.hpc:
                print_ks_states(self.run_loc)

        else:
            print("aims.out file found in ground calculation directory")
            print("skipping calculation...")


def get_element_symbols(geom, spec_at_constr) -> List[str]:
    """
    Find the element symbols from specified atom indices in a geometry file.

    Parameters
    ----------
        geom : str
            path to the geometry file
        spec_at_constr : list
            list of atom indices

    Returns
    -------
        element_symbols : List[str]
            list of element symbols
    """

    with open(geom, "r") as geom:
        lines = geom.readlines()

    atom_lines = []

    # Copy only the lines which specify atom coors into a new list
    for line in lines:
        spl = line.split()
        if len(line) > 0 and "atom" == spl[0]:
            atom_lines.append(line)

    element_symbols = []

    # Get the element symbols from the atom coors
    # Uniquely add each element symbol
    for atom in spec_at_constr:
        element = atom_lines[atom].split()[-1]

        if element not in element_symbols:
            element_symbols.append(element)

    return element_symbols