import glob
import os
import warnings
from sys import platform
from typing import Any, Literal

import numpy as np
import yaml
from ase import Atoms
from ase.build import molecule
from ase.calculators.aims import Aims
from ase.data.pubchem import pubchem_atoms_search
from ase.io import write
from click import BadParameter, File, MissingParameter, progressbar

import deltascf_aims.force_occupation as fo


def add_control_opts(
    start,
    constr_atom: str,
    i_atom: int | str,
    calc: str,
    control_opts: dict,
) -> None:
    """
    Add additional control options to the control file.

    Parameters
    ----------
    start
        Instance of Start class
    constr_atoms : str
        Constrained atom
    i_atom : int | str
        Atom index to add the control options to
    calc : str
        Name of the calculation to add the control options to
    control_opts : dict
        Control options
    """
    # Convert non-string array-type structures to strings
    for key, opt in control_opts.items():
        if not isinstance(opt, str):  # Must be list, tuple, or set
            control_opts[key] = " ".join(str(i) for i in opt)

    parsed_control_opts = fo.ForceOccupation.get_control_keywords(
        f"{start.run_loc}/{constr_atom}{i_atom}/{calc}/control.in"
    )
    mod_control_opts = fo.ForceOccupation.mod_keywords(
        control_opts, parsed_control_opts
    )
    control_content = fo.ForceOccupation.change_control_keywords(
        f"{start.run_loc}/{constr_atom}{i_atom}/{calc}/control.in",
        mod_control_opts,
    )

    with open(
        f"{start.run_loc}/{constr_atom}{i_atom}/{calc}/control.in",
        "w",
    ) as control_file:
        control_file.writelines(control_content)


def add_molecule_identifier(
    start, atom_specifier: list[int], basis: bool = False
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


def build_geometry(geometry: str) -> Atoms | list[Atoms]:
    """
    Try getting geometry data from various databases to create a geometry.in file.

    Parameters
    ----------
    geometry : str
        Name or formula of the system to be created

    Returns
    -------
    atoms : Atoms | list[Atoms]
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
        print("molecule found as a PubChem name")
    except ValueError:
        print(f"{geometry} not found in PubChem name")
    else:
        return atoms

    try:
        atoms = pubchem_atoms_search(cid=geometry)
        print("molecule found in PubChem CID")
    except ValueError:
        print(f"{geometry} not found in PubChem CID")
    else:
        return atoms

    try:
        atoms = pubchem_atoms_search(smiles=geometry)
        print("molecule found in PubChem SMILES")
    except ValueError as err:
        print(f"{geometry} not found in PubChem smiles")
        print(f"{geometry} not found in PubChem or ASE database")
        print("aborting...")
        raise SystemExit from err
    else:
        return atoms


def check_args(*args: Any) -> None:
    """
    Check if the required arguments have been specified.

    Parameters
    ----------
    *args: Any
        Arguments to check

    Raises
    ------
    MissingParameter
        A required parameter has not been given
    """
    # TODO: Check that this function is working correctly
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


def check_constrained_geom(geom_file: str) -> None:
    """
    Check for `constrain_relaxation` keywords in the geometry.in.

    Parameters
    ----------
    geom_file : str
        Path to geometry.in file

    Raises
    ------
    SystemExit
        Exit the program if the `constrain_relaxation` keyword is found
    """
    for line in geom_file:
        if "constrain_relaxation" in line:
            print("`constrain_relaxation` keyword found in geometry.in")
            print("ensure that no atoms are fixed in the geometry.in file")
            print(
                "the geometry of the structure should have already been relaxed before "
                "running single-point calculations"
            )
            print("aborting...")
            raise SystemExit(1)


def check_curr_prev_run(
    run_type: Literal["ground", "hole", "init_1", "init_2"],
    run_loc: str,
    constr_atoms: list[str] | str,
    atom_specifier: list[int],
    constr_method: Literal["projector", "basis"],
    hpc: bool,
    force: bool = False,
) -> None:
    """

    Check if the current calculation has previously been run.

    Parameters
    ----------
    run_type : Literal["ground", "hole", "init_1", "init_2"]
        Type of calculation to check for
    run_loc : str
        Path to the calculation directory
    constr_atoms : list[str] | str
        Constrained atoms
    atom_specifier : list[int]
        list of atom indices
    constr_method : Literal["projector", "basis"]
        Method of constraining atom occupations
    hpc : bool
        Whether to run on a HPC
    force : bool, optional
        Force the calculation to run

    Raises
    ------
    SystemExit
        Exit the program if the calculation has already been run
    """
    if run_type == "ground":
        search_path = f"{run_loc}/{run_type}/aims.out"
    elif constr_method == "projector":
        search_path = (
            f"{run_loc}/{constr_atoms[0]}{atom_specifier[0]}/{run_type}/aims.out"
        )
    elif constr_method == "basis":
        search_path = f"{run_loc}/{constr_atoms[0]}{atom_specifier[0]}/aims.out"

    if os.path.isfile(search_path) and not hpc and not force:
        warnings.warn("Calculation has already been completed")
        cont = None
        while cont != "y":
            cont = str(input("Do you want to continue? (y/n) ")).lower()

            if cont == "n":
                print("aborting...")
                raise SystemExit(1)

            if cont == "y":
                break


def check_k_grid(control_file: str) -> bool:
    """
    Check if there is a k_grid in the control.in.

    Parameters
    ----------
    control_file : str
        Path to control.in file

    Returns
    -------
    k_grid : bool
        Whether the k_grid input parameter is found
    """
    k_grid = False

    for line in control_file:
        if "k_grid" in line:
            k_grid = True

    return k_grid


def check_lattice_vecs(geom_file: str) -> bool:
    """
    Check if lattice vectors are given in the geometry.in file.

    Parameters
    ----------
    geom_file : str
        Path to geometry.in file

    Returns
    -------
    l_vecs : bool
        True if lattice vectors are found, False otherwise
    """
    l_vecs = False

    for line in geom_file:
        if "lattice_vector" in line:
            l_vecs = True

    return l_vecs


def check_params(start, include_hpc=False) -> None:
    """
    Check that the parameters given in Start are valid.

    Parameters
    ----------
    start
        Instance of the Start class
    include_hpc : bool, optional
        Include the hpc parameter in the check

    Raises
    ------
    MissingParameter
        A required parameter has not been given
    BadParameter
        An incompatible parameter has been given
    """
    if include_hpc and start.hpc:
        raise BadParameter(
            "the -h/--hpc flag is only supported for the 'hole' run type"
        )

    if len(start.spec_at_constr) == 0 and len(start.constr_atom) == 0:
        raise MissingParameter(
            param_hint="-c/--constrained_atom or -s/--specific_atom_constraint",
            param_type="option",
        )


def check_species_in_control(control_content: list[str], species: str) -> bool:
    """
    Check if the species basis set definition exists in control.in.

    Parameters
    ----------
    control_content : list[str]
        Lines from the control.in file
    species : str
        Element of the basis set to search for

    Returns
    -------
    bool
        True if the basis set for the species was found in control.in, False otherwise

    """
    for line in control_content:
        spl = line.split()

        if len(spl) > 0 and spl[0] == "species" and species == spl[1]:
            return True

    return False


def convert_opts_to_dict(opts: tuple[str], pbc: tuple[int] | None) -> dict:
    """
    Convert the control options from a tuple to a dictionary.

    Parameters
    ----------
    opts : tuple[str]
        tuple of control options
    pbc : tuple[int]
        tuple of k-points

    Returns
    -------
    opts_dict : dict
        Dictionary of control options
    """
    opts_dict = {}

    for opt in opts:
        spl = opt.split(sep="=")

        opts_dict[spl[0]] = spl[1]

    # Also add k_grid if given
    if pbc is not None:
        opts_dict.update({"k_grid": pbc})

    return opts_dict


def convert_tuple_key_to_str(control_opts: dict) -> dict:
    """
    Convert any keys given as tuples to strings in control_opts.

    Parameters
    ----------
    control_opts : dict
        Options for the control.in file

    Returns
    -------
    control_opts : dict
        Ammended control.in file options
    """
    for i in control_opts.items():
        if isinstance(i[1], tuple):
            control_opts[i[0]] = " ".join(str(j) for j in i[1])

    return control_opts


def create_calc(
    procs: int, binary: str, aims_cmd: str, species: str, int_grid: str
) -> Aims:
    """
    Create an ASE calculator object.

    Parameters
    ----------
    procs : int
        number of processors to use
    binary : str
        path to aims binary
    aims_cmd : str
        command to run aims
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
    return Aims(
        xc="pbe",
        spin="collinear",
        default_initial_moment=1,
        aims_command=f"{aims_cmd} {procs} {binary}",
        species_dir=f"{species}/defaults_2020/{int_grid}/",
    )


def get_atoms(
    constr_atoms: list[str] | str,
    spec_at_constr: list[int],
    geometry_path: str,
    element_symbols: str | list[str],
) -> list[int]:
    """
    Get the atom indices to constrain from the geometry file.

    Parameters
    ----------
    constr_atoms : list[str] | str
        list of elements to constrain
    spec_at_constr : list[int]
        list of atom indices to constrain
    geometry_path : str
        Path to the geometry file
    element_symbols : str | list[str]
        Element symbols to constrain

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

    # For if the user supplied element symbols to constrain
    if isinstance(constr_atoms, list):
        # Check validity of specified elements
        for atom in constr_atoms:
            if atom not in elements:
                raise ValueError("invalid element specified")

        print("Calculating all target atoms in geometry.in")

        # Constrain all atoms of the target element
        for atom in constr_atoms:
            with open(geometry_path) as geom_in:
                atom_counter = 0

                for line in geom_in:
                    spl = line.split()

                    if len(spl) > 0 and "atom" in spl[0]:
                        atom_counter += 1
                        element = spl[-1]  # Identify atom
                        identifier = spl[0]  # Extra check that line is an atom

                        if "atom" in identifier and element == atom:
                            atom_specifier.append(atom_counter)

    # For if the user supplied atom indices to constrain
    elif len(spec_at_constr) > 0:
        # Check validity of specified elements
        for atom in element_symbols:
            if atom not in elements:
                raise ValueError("Invalid element specified")

        atom_specifier = list(spec_at_constr)

    else:
        raise MissingParameter(
            param_hint="-c/--constrained_atom or -s/--specific_atom_constraint",
            param_type="option",
        )

    print("Specified atom indices:", atom_specifier)

    return atom_specifier


def get_all_elements() -> list[str]:
    """
    Get a list of all element symbols supported by FHI-aims.

    Returns
    -------
    elements : list[str]
        Element symbols
    """
    # Find the root directory of the package
    current_path = os.path.dirname(os.path.realpath(__file__))

    # Get all supported elements in FHI-aims
    with open(f"{current_path}/elements.yml") as elements_file:
        return yaml.load(elements_file, Loader=yaml.SafeLoader)


def get_element_symbols(geom: str, spec_at_constr: list[int]) -> list[str]:
    """
    Find the element symbols from specified atom indices in a geometry file.

    Parameters
    ----------
    geom : str
        Path to the geometry file
    spec_at_constr : list[int]
        list of atom indices

    Returns
    -------
    list[str]
        list of element symbols
    """
    with open(geom) as geom_file:
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


def _check_spin_polarised(lines: list[str]) -> bool:
    """
    Check if the FHI-aims calculation was spin polarised.

    Parameters
    ----------
    lines : list[str]
        Lines from the aims.out file

    Returns
    -------
    bool
        Whether the calculation was spin polarised or not
    """
    spin_polarised = False

    for line in lines:
        spl = line.split()
        if len(spl) == 2:
            # Don't break the loop if spin polarised calculation is found as if the
            # keyword is specified again, it is the last one that is used
            if spl[0] == "spin" and spl[1] == "collinear":
                spin_polarised = True

            if spl[0] == "spin" and spl[1] == "none":
                spin_polarised = False

    return spin_polarised


def print_ks_states(run_loc: str) -> None:
    """
    Print the Kohn-Sham eigenvalues from a calculation.

    Parameters
    ----------
    run_loc : str
        Path to the calculation directory

    Raises
    ------
    ValueError
        Could not find the KS states
    """
    with open(f"{run_loc}/aims.out") as aims:
        lines = aims.readlines()

    # Check if the calculation was spin polarised
    spin_polarised = _check_spin_polarised(lines)

    # Parse line to find the start of the KS eigenvalues
    target_line = "  State    Occupation    Eigenvalue [Ha]    Eigenvalue [eV]"

    if not spin_polarised:
        start_line, end_line = None, None

        for num, content in enumerate(reversed(lines)):
            if target_line in content:
                start_line = len(lines) - num
                break

        else:
            raise ValueError("Could not find the KS states in aims.out")

        for num, content in enumerate(lines[start_line:]):
            if not content.split():
                end_line = start_line + num
                break

        else:
            raise ValueError("Could not find the KS states in aims.out")

        eigs = lines[start_line:end_line]

        print("\nKS eigenvalues:\n")
        print(target_line)
        print(*eigs, sep="")

    elif spin_polarised:
        su_eigs_start_line, sd_eigs_start_line = None, None

        for num, content in enumerate(reversed(lines)):
            if "Spin-up eigenvalues" in content:
                su_eigs_start_line = len(lines) - num + 1

                if "K-point:" in lines[su_eigs_start_line - 1]:
                    su_eigs_start_line += 1
                break

        else:
            raise ValueError("No spin-up KS states found")

        for num, content in enumerate(lines[su_eigs_start_line:]):
            if not content.split():
                su_eigs_end_line = su_eigs_start_line + num
                break

        else:
            raise ValueError("No spin-up KS states found")

        if "Spin-down eigenvalues" in lines[su_eigs_end_line + 1]:
            sd_eigs_start_line = su_eigs_end_line + 3

            if "K-point:" in lines[su_eigs_end_line + 2]:
                sd_eigs_start_line += 1

        else:
            raise ValueError("No spin-down KS states found")

        for num, content in enumerate(lines[sd_eigs_start_line:]):
            if not content.split():
                sd_eigs_end_line = sd_eigs_start_line + num
                break

        else:
            raise ValueError("No spin-down KS states found")

        su_eigs = lines[su_eigs_start_line:su_eigs_end_line]
        sd_eigs = lines[sd_eigs_start_line:sd_eigs_end_line]

        print("Spin-up KS eigenvalues:\n")
        print(*su_eigs, sep="")
        print("Spin-down KS eigenvalues:\n")
        print(*sd_eigs, sep="")


def set_env_vars() -> None:
    """Set environment variables for running FHI-aims."""
    os.system("export OMP_NUM_THREADS=1")
    os.system("export MKL_NUM_THREADS=1")
    os.system("export MKL_DYNAMIC=FALSE")

    if platform == "linux" or platform == "linux2":
        os.system("ulimit -s unlimited")
    elif platform == "darwin":
        os.system("ulimit -s hard")
    else:
        warnings.warn("OS not supported, please ensure ulimit is set to unlimited")


def warn_no_extra_control_opts(opts: dict, inp: File | None) -> None:
    """
    Raise a warning if not additional control options have been specified.

    Parameters
    ----------
    opts : dict
        additional control options to be added to the control.in file
    inp : File | None
        path to custom control.in file

    """
    if len(opts) < 1 and inp is None:
        warnings.warn(
            "No extra control options provided, using default options which can be "
            "found in the 'control.in' file",
            stacklevel=2,
        )


def write_control(
    run_loc: str,
    control_opts: dict,
    atoms: Atoms,
    int_grid: str,
    add_extra_basis: bool,
    defaults: str,
) -> None:
    """
    Write a control.in file.

    Parameters
    ----------
    run_loc : str
        Path to the calculation directory
    control_opts : dict
        Dictionary of control options
    atoms : Atoms
        ASE atoms object
    int_grid : str
        Basis set density
    add_extra_basis : bool
        True if extra basis functions are to be added to the basis set, False
        otherwise
    defaults : str
        Path to the species_defaults directory
    """
    # Firstly create the control file if it doesn't exist
    if not os.path.isfile(f"{run_loc}/control.in"):
        os.system(f"touch {run_loc}/control.in")

    control_opts = convert_tuple_key_to_str(control_opts)

    # Use the static method from ForceOccupation
    lines = fo.ForceOccupation.change_control_keywords(
        f"{run_loc}/control.in", control_opts
    )

    with open(f"{run_loc}/control.in", "w") as control:
        control.writelines(lines)

    # Then add the basis set
    elements = list(set(atoms.get_chemical_symbols()))

    for el in elements:
        # TODO Add extra basis functions for ground state calculations
        # if add_extra_basis:
        #     basis_set = glob.glob(f"{defaults}/ch_basis_sets/{int_grid}/*{el}_default")[
        #         0
        #     ]

        #     os.system(f"cat {basis_set} >> {run_loc}/control.in")

        if not check_species_in_control(lines, el):
            basis_set = glob.glob(f"{defaults}/defaults_2020/{int_grid}/*{el}_default")[
                0
            ]
            os.system(f"cat {basis_set} >> {run_loc}/control.in")

    # Copy it to the ground directory
    os.system(f"cp {run_loc}/control.in {run_loc}/ground")


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
    setup_ground(geom_inp, control_inp, control_opts, start)
        Setup the ground calculation files and directories
    add_extra_basis_fns(constr_atom)
        Add additional basis functions to the basis set
    run_ground(control_opts, add_extra_basis, l_vecs, print_output, nprocs, binary, calc)
        Run the ground state calculation
    """

    def __init__(self, run_loc, atoms, basis_set, species, ase, hpc):
        self.run_loc = run_loc
        self.atoms = atoms
        self.basis_set = basis_set
        self.species = species
        self.ase = ase
        self.hpc = hpc

    def setup_ground(self, geom_inp, control_inp, control_opts, start) -> None:
        """
        Setup the ground calculation files and directories.

        Parameters
        ----------
        geom_inp : str
            path to the geometry.in file
        control_inp : str
            path to the control.in file
        control_opts : dict
            additional options to be added to the control.in file
        start
            instance of Start class
        """
        # Create the ground directory if it doesn't already exist
        os.system(f"mkdir -p {self.run_loc}/ground")

        # Write the geometry file if the system is specified through CLI
        if geom_inp is None:
            write(f"{self.run_loc}/geometry.in", self.atoms, format="aims")

        # Copy the geometry.in and control.in files to the ground directory
        if control_inp is not None:
            os.system(f"cp {control_inp.name} {self.run_loc}/ground")

            # Add any additional options to the control file
            if len(control_opts) > 0:
                add_control_opts(start, "", "", "ground", control_opts)

        if geom_inp is not None:
            os.system(f"cp {geom_inp.name} {self.run_loc}/ground")

    def add_extra_basis_fns(self, constr_atom, control_in) -> None:
        """
        Add additional basis functions to the basis set.

        Parameters
        ----------
        constr_atom : str
            element symbol of the constrained atom
        """
        current_path = os.path.dirname(os.path.realpath(__file__))
        with open(control_in.name) as control:
            control_content = control.readlines()

        with open(f"{current_path}/elements.yml") as elements:
            elements = yaml.load(elements, Loader=yaml.SafeLoader)

        new_content = fo.ForceOccupation.add_additional_basis(
            elements, control_content, constr_atom
        )

        with open(control_in.name, "w") as control:
            control.writelines(new_content)

    def _with_ase(self, calc, control_opts, add_extra_basis, l_vecs) -> None:
        """
        Run the ground state calculation using ASE.

        Parameters
        ----------
        calc : Aims
            FHI-aims calculator instance
        control_opts : dict
            Control options
        add_extra_basis : bool
            Whether to add extra basis sets for a core hole
        l_vecs : tuple
            Lattice vectors
        """
        # Change the defaults if any are specified by the user
        # Update with all control options from the calculator
        calc.set(**control_opts)
        control_opts = calc.parameters

        if l_vecs is not None:
            # Create a 3x3 array of lattice vectors
            l_vecs_copy = np.zeros((3, 3))
            for i in range(3):
                l_vecs_copy[i] = [int(j) for j in l_vecs[i].split()]

            l_vecs = l_vecs_copy

            # Add the lattice vectors if periodic
            # Currently only supports 3D periodicity
            self.atoms.set_pbc((True, True, True))
            self.atoms.set_cell(l_vecs)

        if self.hpc:
            # Prevent species dir from being written
            control_opts.pop("species_dir")

            print("writing geometry.in file...")
            write(
                f"{self.run_loc}/ground/geometry.in", images=self.atoms, format="aims"
            )

            print("writing control.in file...")
            write_control(
                self.run_loc,
                control_opts,
                self.atoms,
                self.basis_set,
                add_extra_basis,
                self.species,
            )

        else:
            print("running calculation...")
            self.atoms.get_potential_energy()
            # Move files to ground directory
            os.system(
                f"cp {self.run_loc}/geometry.in {self.run_loc}/control.in {self.run_loc}/ground/"
            )
            os.system(
                f"mv {self.run_loc}/aims.out {self.run_loc}/parameters.ase {self.run_loc}/ground/"
            )

    def _without_ase(self, print_output, aims_cmd, nprocs, binary) -> None:
        """
        Run the ground state calculation without ASE.

        Parameters
        ----------
        print_output : bool
            Whether to print the output of the calculation
        aims_cmd : str
            Command to run FHI-aims
        nprocs : int
            Number of processors to use with parallel command
        binary : str
            Path to the FHI-aims binary
        """
        print("running calculation...")

        if print_output:  # Show live output of calculation
            os.system(
                f"cd {self.run_loc}/ground && {aims_cmd} {nprocs} {binary} | tee aims.out"
            )

        else:
            os.system(
                f"cd {self.run_loc}/ground && {aims_cmd} {nprocs} {binary} > aims.out"
            )

    def run_ground(
        self,
        control_opts,
        add_extra_basis,
        l_vecs,
        print_output,
        aims_cmd,
        nprocs,
        binary,
        calc=None,
    ) -> None:
        """
        Run the ground state calculation.

        Parameters
        ----------
        control_opts : dict
            Control options
        add_extra_basis : bool
            Whether to add additional basis function to the core hole
        l_vecs :
        tuple[str] | None
            Lattice vectors
        print_output : bool
            Whether to print the output of the calculation
        aims_cmd : str
            Command to run FHI-aims
        nprocs : int
            Number of processors to use with the parallel command
        binary : str
            Path to the FHI-aims binary
        calc : Aims, optional
            Instance of an ASE calculator object
        """
        if not self.hpc:  # Run the ground state calculation
            set_env_vars()

        if self.ase:  # Use ASE
            self._with_ase(calc, control_opts, add_extra_basis, l_vecs)

        elif not self.hpc:  # Don't use ASE
            self._without_ase(print_output, aims_cmd, nprocs, binary)

        # Print the KS states from aims.out so it is easier to specify the
        # KS states for the hole calculation
        if not self.hpc:
            print_ks_states(f"{self.run_loc}/ground/")


class ExcitedCalc:
    """
    Setup and run an excited state calculation.

    Attributes
    ----------
    start
        Instance of Start class

    Methods
    -------
    check_restart_files(constr_atoms, prev_calc, atom)
        Check if the restart files from the previous calculation exist
    check_prereq_calc(current_calc, constr_atoms, constr_method)
        Check if the prerequisite calculation has been run
    run_excited(atom_specifier, constr_atoms, run_type, spec_run_info)
        Run an excited state calculation
    """

    def __init__(self, start):
        self.start = start

    def check_restart_files(self, constr_atoms, prev_calc, atom) -> None:
        """
        Check if the restart files from the previous calculation exist.

        Parameters
        ----------
        constr_atoms : list[str]
            Atom indices to constrain
        prev_calc : str
            Name of the previous calculation to check
        atom : int
            Atom index to constrain

        Raises
        ------
        FileNotFoundError
            Unable to find restart files in the directory of the previous
            calculation
        """
        if (
            len(
                glob.glob(
                    f"{self.start.run_loc}/{constr_atoms[0]}"
                    f"{atom}/{prev_calc}/*restart*"
                )
            )
            < 1
        ):
            print(
                f'{prev_calc} restart files not found, please ensure "{prev_calc}"'
                " has been run"
            )
            raise FileNotFoundError

    def check_prereq_calc(
        self,
        current_calc: Literal["init_1", "init_2", "hole"],
        constr_atoms,
        constr_method: Literal["projector", "basis"],
    ) -> Literal["ground", "init_1", "init_2"] | None:
        """
        Check if the prerequisite calculation has been run.

        Parameters
        ----------
        current_calc : Literal["init_1", "init_2", "hole"]
            Type of excited calculation to check for
        constr_atoms : list[str]
            list of constrained atoms
        constr_method : Literal["projector", "basis"]
            Method of constraining atomic core holes

        Returns
        -------
        Literal["ground", "init_1", "init_2"] | None
            Name of the prerequisite calculation, None if prev_calc has not been
            assigned

        Raises
        ------
        FileNotFoundError
            Could not find the aims.out file for the prerequisite calculation
        TypeError
            Invalid type for current_calc has been given
        ValueError
            Invalid parameter for current_calc has been given
        """
        # Placeholders until assigned
        prev_calc = None
        search_path = ""

        match current_calc:
            case "init_1":
                prev_calc = "ground"
                search_path = f"{self.start.run_loc}/ground/aims.out"

            case "init_2":
                prev_calc = "init_1"

                try:
                    search_path = glob.glob(
                        f"{self.start.run_loc}/{constr_atoms[0]}*/init_1/aims.out"
                    )[0]
                except FileNotFoundError:
                    raise FileNotFoundError(
                        f"aims.out for {prev_calc} not found, please ensure the "
                        f"{prev_calc} calculation has been run"
                    )

            case "hole":
                if constr_method == "projector" and not self.start.hpc:
                    prev_calc = "init_2"

                    try:
                        search_path = glob.glob(
                            f"{self.start.run_loc}/{constr_atoms[0]}*/init_2/aims.out"
                        )[0]
                    except FileNotFoundError:
                        raise FileNotFoundError(
                            f"aims.out for {prev_calc} not found, please ensure the "
                            f"{prev_calc} calculation has been run"
                        )

                if constr_method == "projector" and self.start.hpc:
                    prev_calc = "ground"
                    search_path = f"{self.start.run_loc}/ground/aims.out"

                if constr_method == "basis":
                    prev_calc = "ground"
                    search_path = f"{self.start.run_loc}/ground/aims.out"

        if not os.path.isfile(search_path):
            raise FileNotFoundError(
                f"aims.out for {prev_calc} not found, please ensure the "
                f"{prev_calc} calculation has been run"
            )

        return prev_calc

    def run_excited(
        self,
        atom_specifier,
        constr_atoms,
        run_type: Literal["init_1", "init_2", "hole", ""],
        spec_run_info,
        basis_constr=False,
    ) -> None:
        """
        Run an excited state calculation.

        Parameters
        ----------
        atom_specifier : list[int]
            list of atom indices to constrain
        constr_atoms : list[str]
            Constrained atoms
        run_type : Literal["init_1", "init_2", "hole", ""]
            Type of excited calculation to run
        spec_run_info : str
            Redirection location for STDERR of calculation
        basis_constr : bool, optional
            Whether the calculation uses the basis occupation constraint method
        """
        # Don't cd into hole for basis calculation
        if basis_constr:
            run_type = ""

        set_env_vars()

        if self.start.print_output:  # Print live output of calculation
            for i in range(len(atom_specifier)):
                os.system(
                    f"cd {self.start.run_loc}/{constr_atoms[0]}{atom_specifier[i]}"
                    f"/{run_type} && {self.start.run_cmd} {self.start.nprocs} "
                    f"{self.start.binary} | tee aims.out {spec_run_info}"
                )
                if run_type != "init_1" and run_type != "init_2":
                    print_ks_states(
                        f"{self.start.run_loc}{constr_atoms[0]}{atom_specifier[i]}/{run_type}/"
                    )

        else:
            with progressbar(
                range(len(atom_specifier)),
                label=f"calculating {run_type}:",
            ) as prog_bar:
                for i in prog_bar:
                    os.system(
                        f"cd {self.start.run_loc}/{constr_atoms[0]}{atom_specifier[i]}"
                        f"/{run_type} && {self.start.run_cmd} {self.start.nprocs} "
                        f"{self.start.binary} > aims.out {spec_run_info}"
                    )

            # TODO figure out how to parse STDOUT so a completed successfully calculation
            # message can be given or not
            # print(f"{run_type} calculations completed successfully")
