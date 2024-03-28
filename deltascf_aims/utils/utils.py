import glob
import os
import warnings
from sys import platform
from typing import List, Literal, Union

import numpy as np
import yaml
from ase import Atoms
from ase.build import molecule
from ase.calculators.aims import Aims
from ase.data.pubchem import pubchem_atoms_search
from ase.io import write
from click import BadParameter, MissingParameter, progressbar

import deltascf_aims.force_occupation as fo


def add_control_opts(
    start,
    constr_atom,
    i_atom,
    calc,
    control_opts,
) -> None:
    """
    Add additional control options to the control file.

    Parameters
    ----------
    start : Start
        Instance of Start class
    constr_atoms : str
        Constrained atom
    i_atom : int
        Atom index to add the control options to
    calc : str
        Name of the calculation to add the control options to
    control_opts : dict
        Control options
    """

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


def add_molecule_identifier(start, atom_specifier, basis=False) -> None:
    """
    Add a string to the geometry.in to parse when plotting to identify it.

    Parameters
    ----------
    start : Start
        Instance of the Start class
    atom_specifier : List[int]
        Atom indices as given in geometry.in
    basis : bool, optional
        Whether a basis calculation is being run
    """

    if basis:
        hole = ""
    else:
        hole = "/hole"

    with open(
        f"{start.run_loc}/{start.constr_atom[0]}{atom_specifier[0]}{hole}/geometry.in",
        "r",
    ) as hole_geom:
        lines = hole_geom.readlines()

    # Check that the molecule identifier is not already in the file
    for line in lines:
        if start.spec_mol in line:
            return

    lines.insert(4, f"# {start.spec_mol}\n")

    with open(
        f"{start.run_loc}/{start.constr_atom[0]}{atom_specifier[0]}{hole}/geometry.in",
        "w",
    ) as hole_geom:
        hole_geom.writelines(lines)


def build_geometry(geometry) -> Union[Atoms, List[Atoms]]:
    """
    Try getting geometry data from various databases to create a geometry.in file.

    Parameters
    ----------
    geometry : str
        Name or formula of the system to be created

    Returns
    -------
    atoms : Union[Atoms, List[Atoms]]
        Atoms object, or list of atoms objects

    Raises
    ------
    SystemExit
        Exit the program if the system is not found in any database
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
    """Check if the required arguments have been specified.

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


def check_constrained_geom(geom_file) -> None:
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
            print("'constrain_relaxation' keyword found in geometry.in")
            print("ensure that no atoms are fixed in the geometry.in file")
            print(
                "the geometry of the structure should have already been relaxed before "
                "running single-point calculations"
            )
            print("aborting...")
            raise SystemExit(1)


def check_curr_prev_run(
    run_type: Literal["ground", "hole", "init_1", "init_2"],
    run_loc,
    constr_atoms,
    atom_specifier,
    constr_method: Literal["projector", "basis"],
    hpc,
) -> None:
    """

    Check if the current calculation has previously been run.

    Parameters
    ----------
    run_type : Literal["ground", "hole", "init_1", "init_2"]
        Type of calculation to check for
    run_loc : str
        Path to the calculation directory
    constr_atoms : List[str]
        Constrained atoms
    atom_specifier : List[int]
        List of atom indices
    constr_method : Literal["projector", "basis"]
        Method of constraining atom occupations
    hpc : bool
        Whether to run on a HPC

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

    if os.path.isfile(search_path) and not hpc:
        warnings.warn("Calculation has already been completed")
        cont = None
        while cont != "y":
            cont = str(input("Do you want to continue? (y/n) ")).lower()

            if cont == "n":
                print("aborting...")
                raise SystemExit(1)

            if cont == "y":
                break


def check_k_grid(control_file) -> bool:
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


def check_lattice_vecs(geom_file) -> bool:
    """
    Check if lattice vectors are given in the geometry.in file.

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


def check_params(start, include_hpc=True) -> None:
    """
    Check that the parameters given in Start are valid.

    Parameters
    ----------
    start : Start
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

    if include_hpc:
        if start.hpc:
            raise BadParameter(
                "the -h/--hpc flag is only supported for the 'hole' run type"
            )

    if len(start.spec_at_constr) == 0 and len(start.constr_atom) == 0:
        raise MissingParameter(
            param_hint="-c/--constrained_atom or -s/--specific_atom_constraint",
            param_type="option",
        )


def check_species_in_control(control_content, species) -> bool:
    """
    Check if the species basis set definition exists in control.in.

    Parameters
    ----------
    control_content : List[str]
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

        if len(spl) > 0 and "species" == spl[0] and species == spl[1]:
            return True

    return False


def convert_opts_to_dict(opts, pbc) -> dict:
    """
    Convert the control options from a tuple to a dictionary.

    Parameters
    ----------
    opts : tuple
        Tuple of control options
    pbc : list
        Tuple of k-points

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


def convert_tuple_key_to_str(control_opts) -> dict:
    """Convert any keys given as tuples to strings in control_opts

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
        if type(i[1]) == tuple:
            control_opts[i[0]] = " ".join(str(j) for j in i[1])

    return control_opts


def create_calc(procs, binary, species, int_grid) -> Aims:
    """
    Create an ASE calculator object.

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


def get_atoms(
    constr_atoms, spec_at_constr, geometry_path, element_symbols
) -> List[int]:
    """
    Get the atom indices to constrain from the geometry file.

    Parameters
    ----------
    constr_atoms : List[str]
        List of elements to constrain
    spec_at_constr : List[int]
        List of atom indices to constrain
    geometry_path : str
        Path to the geometry file
    element_symbols : Union[str, List[str]]
        Element symbols to constrain

    Returns
    -------
    atom_specifier : List[int]
        List of atom indices to constrain

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
    if constr_atoms is not None:
        # Check validity of specified elements
        for atom in constr_atoms:
            if atom not in elements:
                raise ValueError("invalid element specified")

        print("Calculating all target atoms in geometry.in")

        # Constrain all atoms of the target element
        for atom in constr_atoms:
            with open(geometry_path, "r") as geom_in:
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


def get_all_elements() -> List[str]:
    """
    Get a list of all element symbols supported by FHI-aims.

    Returns
    -------
    elements : List[str]
        Element symbols
    """

    # Find the root directory of the package
    current_path = os.path.dirname(os.path.realpath(__file__))

    # Get all supported elements in FHI-aims
    with open(f"{current_path}/../elements.yml", "r") as elements_file:
        elements = yaml.load(elements_file, Loader=yaml.SafeLoader)

    return elements


def get_element_symbols(geom, spec_at_constr) -> List[str]:
    """
    Find the element symbols from specified atom indices in a geometry file.

    Parameters
    ----------
    geom : str
        Path to the geometry file
    spec_at_constr : List[int]
        List of atom indices

    Returns
    -------
    List[str]
        List of element symbols
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


def print_ks_states(run_loc) -> None:
    """
    Print the KS states for the different spin states.

    Parameters
    ----------
    run_loc : str
        path to the calculation directory

    Raises
    ------
    SystemExit
        Exit the program if no KS states are found
    """

    # Parse the output file
    with open(f"{run_loc}/aims.out", "r") as aims:
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
    if lines[-2] == "          Have a nice day.\n":
        if su_eigs_start_line is None:
            print("No spin-up KS states found")
            print("Did you run a spin polarised calculation?")
            raise SystemExit

        if sd_eigs_start_line is None:
            print("No spin-down KS states found")
            print("Did you run a spin polarised calculation?")
            raise SystemExit

    su_eigs = []
    sd_eigs = []

    # Save the KS states into lists
    if su_eigs_start_line is not None:
        for num, content in enumerate(lines[su_eigs_start_line + 2 :]):
            spl = content.split()

            if len(spl) != 0:
                su_eigs.append(content)
            else:
                break

    if sd_eigs_start_line is not None:
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


def set_env_vars() -> None:
    """
    Set environment variables for running FHI-aims.
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


def warn_no_extra_control_opts(opts, inp) -> None:
    """
    Raise a warning if not additional control options have been specified.

    Parameters
    ----------
    opts : Tuple[str]
        additional control options to be added to the control.in file
    inp : click.File
        path to custom control.in file

    """
    if len(opts) < 1 and inp is None:
        warnings.warn(
            "No extra control options provided, using default options which can be "
            "found in the 'control.in' file"
        )


def write_control(
    run_loc, control_opts, atoms, int_grid, add_extra_basis, defaults
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
        start : Start
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

    def add_extra_basis_fns(self, constr_atom) -> None:
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

        with open(f"{current_path}/../deltascf_aims/elements.yml", "r") as elements:
            elements = yaml.load(elements, Loader=yaml.SafeLoader)

        new_content = fo.ForceOccupation.add_additional_basis(
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

    def _with_ase(self, calc, control_opts, add_extra_basis, l_vecs) -> None:
        """Run the ground state calculation using ASE.

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
            self.atoms.set_pbc(l_vecs)
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

    def _without_ase(self, print_output, nprocs, binary) -> None:
        """

        Run the ground state calculation without ASE.

        Parameters
        ----------
        print_output : bool
            Whether to print the output of the calculation
        nprocs : int
            Number of processors to use with mpirun
        binary : str
            Path to the FHI-aims binary
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

    def run_ground(
        self,
        control_opts,
        add_extra_basis,
        l_vecs,
        print_output,
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
        l_vecs : Union[Tuple[str], None]
            Lattice vectors
        print_output : bool
            Whether to print the output of the calculation
        nprocs : int
            Number of processors to use with mpirun
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
            self._without_ase(print_output, nprocs, binary)

        # Print the KS states from aims.out so it is easier to specify the
        # KS states for the hole calculation
        if not self.hpc:
            print_ks_states(f"{self.run_loc}/ground/")


class ExcitedCalc:
    """
    Setup and run an excited state calculation.

    Attributes
    ----------
    start : Start
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
        constr_atoms : List[str]
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
                "has been run"
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
        constr_atoms : List[str]
            List of constrained atoms
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

        prev_calc = None  # Placeholder until prev_calc is assigned

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
                            f"{self.start.run_loc}/{constr_atoms[0]}*/init_2/"
                            "aims.out"
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

            case _:
                if isinstance(current_calc, str):
                    raise ValueError(
                        "current_calc must be 'init_1', 'init_2', or 'hole', not "
                        f"{current_calc}"
                    )
                else:
                    raise TypeError(
                        "current_calc must be a string with a value of init_1, init_2, "
                        f"or hole, not {type(current_calc)}"
                    )

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
        atom_specifier : List[int]
            List of atom indices to constrain
        constr_atoms : List[str]
            Constrained atoms
        run_type : Literal["init_1", "init_2", "hole"]
            Type of excited calculation to run
        spec_run_info : str
            Redirection location for STDERR of calculation
        basis_constr : bool, optional
            Whether the calculation uses the basis occupation constraint method
        """

        # Don't cd into hole for basis scalculation
        if basis_constr:
            run_type = ""

        set_env_vars()

        if self.start.print_output:  # Print live output of calculation
            for i in range(len(atom_specifier)):
                os.system(
                    f"cd {self.start.run_loc}/{constr_atoms[0]}{atom_specifier[i]}"
                    f"/{run_type} && mpirun -n {self.start.nprocs} "
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
                        f"/{run_type} && mpirun -n {self.start.nprocs} "
                        f"{self.start.binary} > aims.out {spec_run_info}"
                    )

            # TODO figure out how to parse STDOUT so a completed successfully calculation
            # message can be given or not
            # print(f"{run_type} calculations completed successfully")
