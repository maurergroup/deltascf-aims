import glob
import os
import warnings
from pathlib import Path
from typing import Literal, Union

import click
import numpy as np
from ase import Atoms
from ase.io import read

import deltascf_aims.calc_dscf as cds
from deltascf_aims import force_occupation
from deltascf_aims.plot import XPSSpectrum
from deltascf_aims.schmid_pseudo_voigt import broaden
from deltascf_aims.utils import utils
from deltascf_aims.utils.utils import ExcitedCalc, GroundCalc


class Start:
    """
    Perform initial checks and setup for running calcultions.

    ...

    Attributes
    ----------
        hpc : bool
            setup a calculation primarily for use on a HPC cluster WITHOUT running the
            calculation
        spec_mol : str
            molecule to be used in the calculation
        geometry_input : click.File()
            specify a custom geometry.in instead of using a structure from PubChem or ASE
        control_input : click.File()
            specify a custom control.in instead of automatically generating one
        binary : bool
            modify the path to the FHI-aims binary
        run_location : click.Path(file_okay=False, dir_okay=True)
            optionally specify a custom location to run the calculation
        constr_atom : str
            atom to be constrained
        spec_at_constr : click.IntRange(min=1, max_open=True)
            atom to constrain; constrain all atoms of this element
        occupation : float
            occupation of the core hole
        n_atoms : click.IntRange(1)
            number of atoms to constrain per calculation
        basis_set : click.choice(['light', 'intermediate', 'tight', 'really_tight'])
            the basis set to use for the calculation
        use_additional_basis : bool
            whether to use additional basis functions
        graph : bool
            print the simulated XPS spectrum
        print_output : bool
            print the live output of the calculation
        run_cmd : str
            parallel command to use for the calculation
        nprocs : int
            number of processors to use
        ase : bool
            whether to use the ASE backend
        atoms : Atoms
            ASE atoms object
        lattice_vecs : bool
            whether lattice vectors are present in the geometry file
        found_k_grid : bool
            whether a k_grid is present in the control file

    Methods
    -------
        check_for_help_arg()
            Print click help if --help flag is given
        check_for_geometry_input()
            Check that the geometry file parameter has been given
        check_for_pbcs()
            Check for lattice vectors and k_grid in input files
        check_ase_usage()
            Check whether ASE should be used or not
        create_structure(ase)
            Initialise an ASE atoms object
        find_constr_atom_element(atoms)
            Find the element of the atom to perform XPS/NEXAFS for
        check_for_bin()
            Check if a binary is saved in ./aims_bin_loc.txt
        bin_path_prompt(current_path, bin_path)
            Ensure the user has entered the path to the binary
        check_species_path(binary)
            Check if the species_defaults directory exists in the correct location
        atoms()
            property method to return the ASE atoms object
        atoms(atoms)
            setter method to set the ASE atoms object
        add_calc(atoms, binary)
            Add an ASE calculator to an Atoms object
    """

    def __init__(
        self,
        hpc,
        spec_mol,
        geometry_input,
        control_input,
        binary,
        run_location,
        constr_atom,
        spec_at_constr,
        occupation,
        n_atoms,
        basis_set,
        use_extra_basis,
        print_output,
        force,
        run_cmd,
        nprocs,
    ):
        self.hpc = hpc
        self.spec_mol = spec_mol
        self.geometry_input = geometry_input
        self.control_input = control_input
        self.binary = binary
        self.run_loc = run_location
        self.constr_atom = constr_atom
        self.occupation = occupation
        self.n_atoms = n_atoms
        self.spec_at_constr = spec_at_constr
        self.basis_set = basis_set
        self.use_extra_basis = use_extra_basis
        self.print_output = print_output
        self.force = force
        self.run_cmd = run_cmd
        self.nprocs = nprocs

        self.ase = True

    # TODO figure out how to get this to work
    # def _check_help_arg(func):
    #     """
    #     Print click help if --help flag is given anywhere in the CLI arguments.
    #     """

    #     @functools.wraps(func)
    #     def wrapper_check_help_arg(self, *args, **kwargs):
    #         if "--help" in sys.argv:
    #             click.help_option()
    #         else:
    #             func(self, *args, **kwargs)

    #         return func(self, *args, **kwargs)

    #     return wrapper_check_help_arg

    def check_for_geometry_input(self) -> None:
        """Check that the geometry file parameter has been given.

        Raises
        ------
        click.MissingParameter
            The param_hint option has not been provided

        """
        if not self.geometry_input:
            raise click.MissingParameter(
                param_hint="-e/--geometry_input", param_type="option"
            )

    def check_for_pbcs(self) -> None:
        """Check for lattice vectors and k-grid in input file."""
        self.found_l_vecs = False
        if self.geometry_input is not None:
            utils.check_constrained_geom(self.geometry_input)
            self.found_l_vecs = utils.check_lattice_vecs(self.geometry_input)
        else:
            self.found_l_vecs = False

        self.found_k_grid = False
        if self.control_input is not None:
            self.found_k_grid = utils.check_k_grid(self.control_input)
        else:
            self.found_k_grid = False

    def check_constr_keywords(self) -> None:
        """
        Ensure that constr_atom or spec_at_constr are given.

        Raises
        ------
        click.MissingParameter
            The param_hint option has not been provided
        """
        # Add 1 to the second value in the tuple in place
        try:
            self.spec_at_constr = (self.spec_at_constr[0], self.spec_at_constr[1] + 1)
            self.spec_at_constr = list(range(*self.spec_at_constr))
        except TypeError:
            self.spec_at_constr = ()

        if self.constr_atom is None and len(self.spec_at_constr) == 0:
            raise click.MissingParameter(
                param_hint="-c/--constrained_atom or -s/--specific_atom_constraint",
                param_type="option",
            )

    def check_ase_usage(self) -> None:
        """Check whether ASE should be used or not."""
        if self.control_input is not None:
            self.ase = False  # Do not use if control.in is specified

    # @_check_help_arg
    def create_structure(self) -> Union[Atoms, list[Atoms]]:
        """
        Initialise an ASE atoms object from geometry.in or an ASE database.

        Returns
        -------
        Union[Atoms, list[Atoms]]
            ASE atoms object

        Raises
        ------
        click.MissingParameter
            The param_hint option has not been provided
        """
        self.atoms = Atoms()

        # Find the structure if not given
        if self.spec_mol is None and self.geometry_input is None:
            try:
                self.atoms = read(f"./{self.run_loc}/ground/geometry.in")
                print(
                    "molecule argument not provided, defaulting to using existing "
                    "geometry.in file from the ground state calculation"
                )

            except FileNotFoundError as err:
                raise click.MissingParameter(
                    param_hint="-m/--molecule or -e/--geometry_input",
                    param_type="option",
                ) from err

        # Build the structure if given
        elif self.ase:
            if self.spec_mol is not None:
                self.atoms = utils.build_geometry(self.spec_mol)
            if self.geometry_input is not None:
                self.atoms = read(self.geometry_input.name)

        return self.atoms

    def find_constr_atom_element(self) -> None:
        """Find the element of the atom to perform XPS/NEXAFS for."""
        # TODO: add support for multiple constrained atoms
        for atom in self.atoms:
            if atom.index in self.spec_at_constr:
                self.constr_atom = atom.symbol
                break

    # @_check_help_arg
    def check_for_bin(self) -> tuple[str, str]:
        """
        Check if a binary is saved in ./aims_bin_loc.txt.

        Returns
        -------
        current_path : str
            path to the current working directory
        bin_path : str
            path to the location of the FHI-aims binary
        """
        current_path = os.path.dirname(os.path.realpath(__file__))

        if Path(f"{current_path}/aims_bin_loc.txt").exists():
            with open(f"{current_path}/aims_bin_loc.txt") as f:
                lines = f.readlines()
        else:
            lines = []

        if len(lines) > 0 and "~" in lines[0][:-1] and not self.binary:
            raise FileNotFoundError(
                "Please provide the full path to the FHI-aims binary"
            )
        try:
            bin_path = lines[0][:-1]
        except IndexError:
            bin_path = ""

        return current_path, bin_path

    def bin_path_prompt(self, current_path: str, bin_path: str) -> str:
        """
        Ensure the user has entered the path to the binary.

        Open the user's $EDITOR to allow them to enter the path.

        Parameters
        ----------
        current_path : str
            path to the current working directory
        bin_path : str
            path to the location of the FHI-aims binary

        Returns
        -------
        binary : str
            path to the location of the FHI-aims binary
        """
        if not Path(bin_path).is_file() or self.binary or bin_path == "":
            marker = (
                "\n# Enter the path to the FHI-aims binary above this line\n"
                "# Ensure that the binary is located in the build directory of FHIaims\n"
                "# and that the full absolute path is provided"
            )
            bin_line = click.edit(marker)
            if bin_line is not None:
                if Path(str(bin_line).split()[0]).exists():
                    with open(f"{current_path}/aims_bin_loc.txt", "w") as f:
                        f.write(bin_line)

                    with open(f"{current_path}/aims_bin_loc.txt") as f:
                        self.binary = f.readlines()[0]

                else:
                    raise FileNotFoundError(
                        "the path given to the FHI-aims binary does not exist"
                    )

            else:
                raise FileNotFoundError(
                    "path to the FHI-aims binary could not be found"
                )

        elif Path(bin_path).exists():
            print(f"specified binary path: {bin_path}")
            self.binary = bin_path

        else:
            raise FileNotFoundError("path to the FHI-aims binary could not be found")

        return self.binary

    def check_species_path(self, binary: str) -> None:
        """
        Check if the species_defaults directory exists in the correct location.

        Parameters
        ----------
        binary : str
            path to the location of the FHI-aims binary
        """
        self.species = f"{Path(binary).parent.parent}/species_defaults/"

        # TODO: check if the warnings module could be used here
        # Check if the species_defaults directory exists in the correct location
        if not Path(self.species).exists():
            print(
                "\nError: ensure the FHI-aims binary is in the 'build' directory of the FHI-aims"
                " source code directory, and that the 'species_defaults' directory exists"
            )
            msg = (
                f"species_defaults directory not found in {Path(binary).parent.parent}"
            )
            raise NotADirectoryError(msg)

    @property
    def atoms(self) -> Atoms:
        """
        ASE atoms object.

        Returns
        -------
        _atoms : Atoms
            ASE atoms object
        """
        return self._atoms

    @atoms.setter
    def atoms(self, atoms: Atoms) -> None:
        """
        Set the ASE atoms object.

        Parameters
        ----------
        atoms : Atoms
            ASE atoms object
        """
        self._atoms = atoms

    def add_calc(self, binary: str) -> Atoms:
        """
        Add an ASE calculator to an Atoms object.

        Parameters
        ----------
        binary : str
            path to the location of the FHI-aims binary

        Returns
        -------
        atoms : Atoms
            ASE atoms object with a calculator added
        """
        self.atoms.calc = utils.create_calc(
            self.nprocs, binary, self.run_cmd, self.species, self.basis_set
        )

        if self.print_output:
            warnings.warn(
                "-p/--print_output is not supported with the ASE backend", stacklevel=2
            )

        return self.atoms


class Process:
    """
    Calculate DSCF values and plot the simulated XPS spectra.

    ...

    Attributes
    ----------
    TODO
    """

    def __init__(
        self,
        start,
        graph,
        intensity=1,
        asym=False,
        a=0.2,
        b=0.0,
        gl_ratio=0.5,
        omega=0.35,
        include_name=True,
        exclude_mabe=False,
        gmp=0.003,
    ):
        self.start = start
        self.graph = graph
        self.intensity = intensity
        self.asym = asym
        self.a = a
        self.b = b
        self.gl_ratio = gl_ratio
        self.omega = omega
        self.include_name = include_name
        self.exclude_mabe = exclude_mabe
        self.gmp = gmp

        # Ensure that the constrained atom(s) have been given
        utils.check_args(("constrained_atom", self.start.constr_atom))

    def calc_dscf_energies(self) -> tuple[list[float], str]:
        """
        Parse absolute energies and calculate deltaSCF energies.

        Returns
        -------
        xps : list[float]
            deltaSCF energies
        """
        grenrgys = cds.read_ground_energy(self.start.run_loc)
        excienrgys, element = cds.read_excited_energy(
            self.start.run_loc, self.start.constr_atom
        )
        xps = cds.calc_delta_scf(self.start.constr_atom, grenrgys, excienrgys)

        return xps, element

    def move_file_to_run_loc(
        self, element: str, file_type: Literal["peaks", "spectrum"]
    ) -> None:
        """
        Move either the peaks or spectrum file to the run location.

        Parameters
        ----------
        element : str
            element the binding energies were calculated for
        file_type : Literal["peaks", "spectrum"]
            type of file to move
        """
        os.system(f"mv {element}_xps_{file_type}.txt {self.start.run_loc}")

    def call_broaden(self, xps) -> np.ndarray:
        """
        Apply pseudo-Voigt peak broadening.

        Parameters
        ----------
        xps : list[float]
            deltaSCF energies

        Returns
        -------
        peaks : np.ndarray
            broadened peaks
        """
        peaks = broaden(
            0,
            1000,
            self.intensity,
            self.gl_ratio,
            xps,
            self.omega,
            self.asym,
            self.a,
            self.b,
        )

        return peaks

    def write_spectrum_to_file(self, peaks, element, bin_width=0.01) -> None:
        """
        Write the spectrum to a text file.

        Parameters
        ----------
        peaks : np.ndarray
            broadened peaks
        element : str
            element the binding energies were calculated for
        bin_width : float
            resolution of the spectral curve - lower values increase the resolution
        """
        data = []
        bin_val = 0.00
        for peak in peaks:
            data.append(f"{bin_val!s} {peak!s}\n")
            bin_val += bin_width

        with open(f"{element}_xps_spectrum.txt", "w") as spec:
            spec.writelines(data)

    def plot_xps(self, xps):
        """
        Plot the XPS spectrum and save as pdf and png files.

        Parameters
        ----------
        xps : list
            list of individual binding energies.
        """
        xps_spec = XPSSpectrum(
            self.gmp, self.start.run_loc, self.start.constr_atom, self.include_name
        )

        print("\nplotting spectrum and calculating MABE...")

        xps_spec.plot(xps, self.exclude_mabe)


class Projector(GroundCalc, ExcitedCalc):
    """
    Force occupation of Kohn-Sham states and project onto basis functions.

    ...

    Attributes
    ----------
    start : Start
        instance of the Start object
    run_type : click.Choice(["ground", "init_1", "init_2", "hole"])
        type of calculation to perform
    occ_type : click.Choice(["deltascf_projector", "force_occupation_projector"])
        use either the refactored or original projector keyword
    pbc : tuple[int]
        k-grid for a periodic calculation
    l_vecs : list[list[float]]
        lattice vectors in a 3x3 matrix of floats
    spin : click.Choice(["1", "2"])
        spin channel of the constraint
    ks_range : click.IntRange(1)
        range of Kohn-Sham states to constrain
    control_opts : tuple[str]
        additional control options to be added to the control.in file

    Methods
    -------
    check_periodic()
        Check if the lattice vectors and k_grid have been provided
    run_ground()
        Run the ground state calculation
    setup_excited_calcs()
        Setup files and parameters required for the initialisation and hole
        calculations
    pre_init_2(fo, atom_specifier)
        Setup everything for the 2nd init calculation
    pre_hole(fo, atom_specifier)
        Setup everything for the hole calculation
    run_excited(start, atom_specifier)
        Run the projector calculations
    """

    def __init__(
        self, start, run_type, occ_type, pbc, l_vecs, spin, ks_range, control_opts
    ):
        # Get methods from GroundCalc
        super().__init__(
            start.run_loc,
            start.atoms,
            start.basis_set,
            start.species,
            start.ase,
            start.hpc,
        )

        # Get methods from ExcitedCalc
        super(GroundCalc, self).__init__(start)

        self.start = start
        self.run_type = run_type
        self.occ_type = occ_type
        self.pbc = pbc
        self.l_vecs = l_vecs
        self.spin = spin
        self.ks_range = ks_range

        self.ground_geom = f"{self.start.run_loc}/ground/geometry.in"
        self.ground_control = f"{self.start.run_loc}/ground/control.in"

        # Convert control options to a dictionary
        self.control_opts = utils.convert_opts_to_dict(control_opts, pbc)

        # Raise a warning if no additional control options have been specified
        utils.warn_no_extra_control_opts(self.control_opts, start.control_input)

        # if isinstance(self.start.spec_at_constr, list):
        #     self.constr_atoms = self.start.spec_at_constr
        if not isinstance(self.start.constr_atom, list):
            # Convert constr_atom to a list
            self.constr_atoms = [self.start.constr_atom]
        else:
            self.constr_atoms = self.start.constr_atom

    def _calc_checks(
        self,
        current_calc: Literal["init_1", "init_2", "hole"],
        check_restart: bool = True,
        check_args: bool = False,
    ) -> None:
        """
        Perform checks before running an excited calculation.

        Parameters
        ----------
        current_calc : Literal["init_1", "init_2", "hole"]
            Type of excited calculation that will be run
        check_restart : bool, optional
            Whether to check for the existance of restart files or not
        check_args : bool, optional
            Whether to check if the required CLI arguments were given or not
        """
        # Check that the previous calculation has been run
        prev_calc = self.check_prereq_calc(current_calc, self.constr_atoms, "projector")

        # Check that the current calculation has not already been run
        utils.check_curr_prev_run(
            self.run_type,
            self.start.run_loc,
            self.constr_atoms,
            self.atom_specifier,
            "projector",
            self.start.hpc,
            self.start.force,
        )

        # Check that the restart files exist from the previous calculation
        if check_restart:
            for i_atom in self.atom_specifier:
                self.check_restart_files(self.constr_atoms, prev_calc, i_atom)

        # Check that the constrained atoms have been given
        if current_calc != "hole":
            utils.check_params(self.start)
        else:
            utils.check_params(self.start, include_hpc=False)

        # Check required arguments have been given
        if check_args:
            utils.check_args(("ks_range", self.ks_range))

    def _call_setups(self, proj: force_occupation.Projector) -> None:
        """
        Set up files and parameters for the initialization and hole calculations.

        Parameters
        ----------
        proj : ForceOccupation.Projector
            Instance of ForceOccupation
        """
        proj.setup_init_1(self.start.basis_set, self.start.species, self.ground_control)
        proj.setup_init_2(
            self.ks_range[0],
            self.ks_range[1],
            self.start.occupation,
            self.occ_type,
            self.spin,
            self.start.found_k_grid,
        )
        proj.setup_hole(
            self.ks_range[0],
            self.ks_range[1],
            self.start.occupation,
            self.occ_type,
            self.spin,
            self.start.found_k_grid,
        )

    def _cp_restart_files(self, atom: int, begin: str, end: str) -> None:
        """
        Copy the restart files from one calculation location to another.

        Parameters
        ----------
        atom : int
            atom to copy the restart files for
        begin : str
            location to copy the restart files from
        end : str
            location to copy the restart files to
        """
        os.path.isfile(
            glob.glob(
                f"{self.start.run_loc}/{self.constr_atoms[0]}{atom}/{begin}/*restart*"
            )[0]
        )
        os.system(
            f"cp {self.start.run_loc}/{self.constr_atoms[0]}{atom}/{begin}/"
            f"*restart* {self.start.run_loc}/{self.constr_atoms[0]}{atom}"
            f"/{end}/"
        )

    def _get_element_symbols(self) -> Union[str, list[str]]:
        """
        Create a list of element symbols to constrain.

        Returns
        -------
        element_symbols : Union[str, list[str]]
            Element symbols to constrain
        """
        if len(self.start.spec_at_constr) > 0:
            element_symbols = utils.get_element_symbols(
                self.ground_geom, self.start.spec_at_constr
            )[0]
            self.constr_atoms = element_symbols
        else:
            element_symbols = self.constr_atoms

        return element_symbols

    def check_periodic(self) -> None:
        """Check if the lattice vectors and k_grid have been provided."""
        print(
            "-p/--pbc argument not given, attempting to use"
            " k_grid from control file or previous calculation"
        )

        # Try to parse the k-grid if other calculations have been run
        try:
            pbc_list = []
            for control in glob.glob(
                f"{self.start.run_loc}/**/control.in", recursive=True
            ):
                with open(control) as control_lines:
                    for line in control_lines:
                        if "k_grid" in line:
                            pbc_list.append(line.split()[1:])

            # If different k_grids have been used for different calculations,
            # then enforce the user to provide the k_grid
            if not pbc_list.count(pbc_list[0]) == len(pbc_list):
                raise click.MissingParameter(param_hint="-p/--pbc", param_type="option")
            pbc_list = tuple([int(i) for i in pbc_list[0]])
            self.control_opts["k_grid"] = pbc_list

        except IndexError as err:
            raise click.MissingParameter(
                param_hint="-p/--pbc", param_type="option"
            ) from err

    def add_l_vecs(self, geom):
        """
        Add lattice vectors to the geometry.in file.

        Parameters
        ----------
        geom : str
            path to the geometry file
        """
        with open(geom) as geom_file:
            geom_content = geom_file.readlines()

        # Check if the lattice vectors are already in the file
        lv_line_1 = None
        for i, line in enumerate(geom_content):
            if "lattice_vector" in line:
                lv_line_1 = i
                break

        # Add lattice vectors to the geometry file
        if lv_line_1 is not None:  # Replace existing lattice vectors
            for i in range(3):
                geom_content[lv_line_1 + i] = (
                    f"lattice_vector {self.l_vecs[i][0]} {self.l_vecs[i][1]} "
                    f"{self.l_vecs[i][2]}\n"
                )

        elif self.start.ase:  # Add after ASE header
            for i in range(5, 8):
                geom_content.insert(
                    i,
                    f"lattice_vector {self.l_vecs[i - 5][0]} {self.l_vecs[i - 5][1]} "
                    f" {self.l_vecs[i - 5][2]}\n",
                )

        else:
            for i in range(3):  # Add to the top of the file
                geom_content.insert(
                    i,
                    f"lattice_vector {self.l_vecs[i][0]} {self.l_vecs[i][1]} "
                    f"{self.l_vecs[i][2]}\n",
                )

        with open(geom, "w") as geom_file:
            geom_file.writelines(geom_content)

    def setup_excited(self) -> tuple[list[int], str]:
        """
        Setup files and parameters required for the init and hole calculations.

        Returns
        -------
            spec_run_info : str
                redirection location for STDERR of calculation
        """
        # Get the element symbols to constrain
        element_symbols = self._get_element_symbols()

        # Get the atom indices to constrain
        self.atom_specifier = utils.get_atoms(
            self.constr_atoms,
            self.start.spec_at_constr,
            self.ground_geom,
            element_symbols,
        )

        self._calc_checks("init_1", check_restart=False, check_args=True)

        # Create the ForceOccupation object
        fo = force_occupation.ForceOccupation(
            element_symbols,
            self.start.run_loc,
            self.ground_geom,
            self.control_opts,
            self.atom_specifier,
            self.start.use_extra_basis,
        )

        # TODO allow this for multiple constrained atoms using n_atoms
        for atom in element_symbols:
            fo.get_electronic_structure(atom)

        proj = force_occupation.Projector(fo)
        self._call_setups(proj)

        spec_run_info = ""

        return self.atom_specifier, spec_run_info

    def pre_init_2(self) -> tuple[list[int], str]:
        """
        Prerequisite before running the 2nd init calculation.

        Returns
        -------
            spec_run_info : str
                Redirection location for STDERR of calculation
        """
        # Get the element symbols to constrain
        element_symbols = self._get_element_symbols()

        # Get the atom indices to constrain
        self.atom_specifier = utils.get_atoms(
            self.constr_atoms,
            self.start.spec_at_constr,
            self.ground_geom,
            element_symbols,
        )

        self._calc_checks("init_2")

        # Add any additional options to the control file
        for i in range(len(self.atom_specifier)):
            if len(self.control_opts) > 0:
                utils.add_control_opts(
                    self.start,
                    self.constr_atoms[0],
                    self.atom_specifier[i],
                    "init_2",
                    self.control_opts,
                )

            # Copy the restart files to init_2 from init_1
            self._cp_restart_files(self.atom_specifier[i], "init_1", "init_2")

        # Prevent SCF not converged errors from printing
        # It could be an issue to do this if any other errors occur
        spec_run_info = " 2>/dev/null"

        return self.atom_specifier, spec_run_info

    def pre_hole(self) -> tuple[list[int], str]:
        """
        Prerequisite to running the hole calculation.

        Returns
        -------
        tuple[list[int], str]
            Indices for atoms as specified in geometry.in, redirection location for
            STDERR of calculation
        """
        # Get element symbols to constrain
        element_symbols = self._get_element_symbols()

        # Get the atom indices to constrain
        self.atom_specifier = utils.get_atoms(
            self.constr_atoms,
            self.start.spec_at_constr,
            self.ground_geom,
            element_symbols,
        )

        # Check for if init_2 hasn't been run
        if not self.start.hpc:
            self._calc_checks("hole")

        if self.start.hpc:
            self._calc_checks("hole", check_restart=False, check_args=True)

            # Create the ForceOccupation object
            fo = force_occupation.ForceOccupation(
                element_symbols,
                self.start.run_loc,
                self.ground_geom,
                self.control_opts,
                self.atom_specifier,
                self.start.use_extra_basis,
            )

            for atom in element_symbols:
                fo.get_electronic_structure(atom)

            # Setup files required for the initialisation and hole calculations
            proj = force_occupation.Projector(fo)
            self._call_setups(proj)

        # Add a tag to the geometry file to identify the molecule name
        if self.start.spec_mol is not None:
            utils.add_molecule_identifier(self.start, self.atom_specifier)

        # Add any additional control options to the hole control file
        for i in range(len(self.atom_specifier)):
            if len(self.control_opts) > 0:
                utils.add_control_opts(
                    self.start,
                    self.constr_atoms[0],
                    self.atom_specifier[i],
                    "hole",
                    self.control_opts,
                )

            # Copy the restart files to hole from init_2
            if not self.start.hpc:
                self._cp_restart_files(self.atom_specifier[i], "init_2", "hole")

        # Don't redirect STDERR to /dev/null as not converged errors should not occur
        # here
        spec_run_info = ""

        return self.atom_specifier, spec_run_info


class Basis(GroundCalc, ExcitedCalc):
    """
    Force occupation of the basis states directly.

    ...

    Attributes
    ----------
    start : Start
        Instance of the Start object
    run_type : click.Choice(["ground", "hole"])
        Type of calculation to perform
    occ_type : click.Choice(["deltascf_basis", "force_occupation_basis"])
        Method of constraining the occupation
    multiplicity : click.Choice(["1", "2"])
        Spin channel of the constraint
    n_qn : int
        Principal quantum number for the basis function to constrain
    l_qn : int
        Angular momentum quantum number for the basis function to constrain
    m_qn : int
        Magnetic quantum number for the basis function to constrain
    ks_max : int
        Highest energy Kohn-Sham state to constrain
    control_opts : tuple[str]
        Additional control options to be added to the control.in file
    ground_geom : str
        Location of the ground state geometry file
    ground_control : str
        Location of the ground state control file
    control_opts : tuple[str]
        Additional control options to be added to the control.in file
    constr_atoms : list[int]
        Atom indices to constrain
    atom_specifier : list[int]
        Atom indices as specified in geometry.in
    """

    def __init__(
        self,
        start,
        run_type,
        occ_type,
        multiplicity,
        n_qn,
        l_qn,
        m_qn,
        ks_max,
        control_opts,
    ):
        super(Basis, self).__init__(
            start.run_loc,
            start.atoms,
            start.basis_set,
            start.species,
            start.ase,
            start.hpc,
        )

        # Get methods from ExcitedCalc
        # Ignore pyright error as I don't think it understands MRO and thinks this calls
        # the GroundCalc __init__
        super(GroundCalc, self).__init__(start)  # pyright: ignore

        self.start = start
        self.run_type = run_type
        self.occ_type = occ_type
        self.multiplicity = multiplicity
        self.n_qn = n_qn
        self.l_qn = l_qn
        self.m_qn = m_qn
        self.ks_max = ks_max

        self.ground_geom = f"{self.start.run_loc}/ground/geometry.in"
        self.ground_control = f"{self.start.run_loc}/ground/control.in"

        # Convert control options to a dictionary
        self.control_opts = utils.convert_opts_to_dict(control_opts, None)

        # Raise a warning if no additional control options have been specified
        utils.warn_no_extra_control_opts(self.control_opts, start.control_input)

        # Convert constr_atoms to a list
        if not isinstance(self.start.constr_atom, list):
            self.constr_atoms = [self.start.constr_atom]
        else:
            self.constr_atoms = self.start.constr_atom

    def _calc_checks(self) -> None:
        """
        Perform checks before running the excited calculation.
        """
        # Check that the current calculation has not already been run
        utils.check_curr_prev_run(
            self.run_type,
            self.start.run_loc,
            self.constr_atoms,
            self.atom_specifier,
            "basis",
            self.start.hpc,
            self.start.force,
        )

        # Check that the constrained atoms have been given
        utils.check_params(self.start, include_hpc=False)

        # Check that the required arguments have been given
        utils.check_args(
            ("ks_max", self.ks_max),
            ("n_qn", self.n_qn),
            ("l_qn", self.l_qn),
            ("m_qn", self.m_qn),
        )

    def _get_element_symbols(self) -> Union[str, list[str]]:
        """
        Create a list of element symbols to constrain

        Returns
        -------
        element_symbols : Union[str, list[str]]
            Element symbols to constrain
        """
        if len(self.start.spec_at_constr) > 0:
            element_symbols = utils.get_element_symbols(
                self.ground_geom, self.start.spec_at_constr
            )[0]
            self.constr_atoms = element_symbols
        else:
            element_symbols = self.constr_atoms

        return element_symbols

    def setup_excited(self) -> list[int]:
        """
        Setup files and parameters required for the hole calculation.

        Returns
        -------
        list[int]
            Indices for atoms as specified in geometry.in
        """
        # Do this outside of _calc_checks as atom_specifier is needed for that function
        self.check_prereq_calc("hole", self.constr_atoms, "basis")

        # Get the element symbols to constrain
        element_symbols = self._get_element_symbols()

        # Get the atom indices to constrain
        self.atom_specifier = utils.get_atoms(
            self.constr_atoms,
            self.start.spec_at_constr,
            self.ground_geom,
            element_symbols,
        )

        self._calc_checks()

        # Create the directories required for the hole calculation
        fo = force_occupation.ForceOccupation(
            element_symbols,
            self.start.run_loc,
            self.ground_geom,
            self.control_opts,
            self.atom_specifier,
            self.start.use_extra_basis,
        )

        basis = force_occupation.Basis(fo)
        basis.setup_basis(
            self.multiplicity,
            self.n_qn,
            self.l_qn,
            self.m_qn,
            self.start.occupation,
            self.ks_max,
            self.occ_type,
            self.start.basis_set,
            self.start.species,
        )

        # Add molecule ID to geometry file
        if self.start.spec_mol is not None:
            utils.add_molecule_identifier(self.start, self.atom_specifier, basis=True)

        return self.atom_specifier
