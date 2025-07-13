import shutil
import warnings
from pathlib import Path
from typing import Literal

import click
import numpy as np
import numpy.typing as npt
from ase import Atoms
from ase.io import read

import deltascf_aims.calc_dscf as cds
from deltascf_aims import force_occupation
from deltascf_aims.plot import XPSSpectrum
from deltascf_aims.schmid_pseudo_voigt import broaden
from deltascf_aims.utils import (
    calculations_utils,
    checks_utils,
    control_utils,
    geometry_utils,
)


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
            specify a custom geometry.in instead of using a structure from PubChem or
            ASE
        control_input : click.File()
            specify a custom control.in instead of automatically generating one
        binary : bool
            modify the path to the FHI-aims binary
        run_location : click.Path(file_okay=False, dir_okay=True)
            optionally specify a custom location to run the calculation
        constr_atom : str
            atom to be constrained
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

    def __init__(  # noqa: PLR0913
        self,
        hpc: bool,
        spec_mol: str,
        geometry_input: Path,
        control_input: Path,
        binary: bool,
        run_location: Path,
        constr_atom: str,
        occupation: float,
        n_atoms: int,
        basis_set: str,
        use_extra_basis: bool,
        print_output: bool,
        force: bool,
        run_cmd: str,
        nprocs: int,
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
        if not self.geometry_input and not self.spec_mol:
            raise click.MissingParameter(
                param_hint="-e/--geometry_input or -m/--molecule", param_type="option"
            )

    def check_for_pbcs(self) -> None:
        """Check for lattice vectors and k-grid in input file."""
        self.found_l_vecs = False
        if self.geometry_input is not None:
            checks_utils.check_constrained_geom(self.geometry_input)
            self.found_l_vecs = checks_utils.check_lattice_vecs(self.geometry_input)
        else:
            self.found_l_vecs = False

        self.found_k_grid = False
        if self.control_input is not None:
            self.found_k_grid = checks_utils.check_k_grid(self.control_input)
        else:
            self.found_k_grid = False

    def check_constr_keywords(self) -> None:
        """
        Ensure that constr_atom is given.

        Raises
        ------
        click.MissingParameter
            The param_hint option has not been provided
        """
        if self.constr_atom is None:  # and len(self.spec_at_constr) == 0:
            raise click.MissingParameter(
                param_hint="-c/--constrained_atom", param_type="option"
            )

    def check_ase_usage(self) -> None:
        """Check whether ASE should be used or not."""
        if self.control_input is not None:
            self.ase = False  # Do not use if control.in is specified

    # @_check_help_arg
    def create_structure(self) -> Atoms:
        """
        Initialise an ASE atoms object from geometry.in or an ASE database.

        Returns
        -------
        ase.Atoms
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
                self.atoms = read(self.run_loc / "ground" / "geometry.in", index=-1)  # pyright: ignore[reportAttributeAccessIssue]
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
                self.atoms = geometry_utils.build_geometry(self.spec_mol)
            if self.geometry_input is not None:
                self.atoms = read(self.geometry_input.name, index=-1)  # pyright: ignore[reportAttributeAccessIssue]

        return self.atoms  # pyright: ignore[reportReturnType]

    # @_check_help_arg
    def check_for_bin(self) -> tuple[Path, Path]:
        """
        Check if a binary is saved in ./aims_bin_loc.txt.

        Returns
        -------
        current_path : pathlib.Path
            path to the current working directory
        bin_path : pathlib.Path
            path to the location of the FHI-aims binary
        """
        current_path = Path(__file__).resolve().parent

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

        return current_path, Path(bin_path)

    def bin_path_prompt(self, current_path: Path, bin_path: Path) -> Path:
        """
        Ensure the user has entered the path to the binary.

        Open the user's $EDITOR to allow them to enter the path.

        Parameters
        ----------
        current_path : pathlib.Path
            path to the current working directory
        bin_path : pathlib.Path
            path to the location of the FHI-aims binary

        Returns
        -------
        binary : pathlib.Path
            path to the location of the FHI-aims binary
        """
        if not bin_path.is_file() or self.binary:
            marker = (
                "\n# Enter the path to the FHI-aims binary above this line\n"
                "# Ensure that the binary is located in the build directory of FHIaims\n"  # noqa: E501
                "# and that the full absolute path is provided"
            )
            bin_line = click.edit(marker)
            if bin_line is not None:
                if Path(bin_line.split()[0]).is_file():
                    with current_path.joinpath("aims_bin_loc.txt").open("w") as f:
                        f.write(str(bin_line))

                    with current_path.joinpath("aims_bin_loc.txt").open() as f:
                        self.binary = Path(f.readlines()[0])

                else:
                    raise FileNotFoundError(
                        "the path given to the FHI-aims binary does not exist"
                    )

            else:
                raise FileNotFoundError(
                    "path to the FHI-aims binary could not be found"
                )

        elif bin_path.is_file():
            print(f"specified binary path: {bin_path}")
            self.binary = bin_path

        else:
            raise FileNotFoundError("path to the FHI-aims binary could not be found")

        return self.binary

    def check_species_path(self, binary: Path) -> None:
        """
        Check if the species_defaults directory exists in the correct location.

        Parameters
        ----------
        binary : pathlib.Path
            path to the location of the FHI-aims binary
        """
        self.species = binary.parent.parent / "species_defaults"

        # TODO: check if the warnings module could be used here
        # Check if the species_defaults directory exists in the correct location
        if not self.species.is_dir():
            print(
                "\nError: ensure the FHI-aims binary is in the `build` directory of the"
                " FHI-aims source code directory, and that the `species_defaults` "
                "directory exists"
            )
            msg = f"species_defaults directory not found in {binary.parent.parent}"
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

    def add_calc(self, binary: Path) -> Atoms:
        """
        Add an ASE calculator to an Atoms object.

        Parameters
        ----------
        binary : Path
            path to the location of the FHI-aims binary

        Returns
        -------
        atoms : Atoms
            ASE atoms object with a calculator added
        """
        self.atoms.calc = calculations_utils.create_calc(
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

    Parameters
    ----------
    start : Start
        instance of the Start object containing initial setup and configuration
    graph : bool
        whether to generate and display a graphical plot of the XPS spectrum
    intensity : float, default=1
        peak intensity scaling factor for the spectrum
    asym : bool, default=False
        whether to apply asymmetric peak broadening
    a : float, default=0.2
        asymmetry parameter for peak broadening
    b : float, default=0.0
        secondary asymmetry parameter for peak broadening
    gl_ratio : float, default=0.5
        Gaussian-Lorentzian mixing ratio for pseudo-Voigt broadening
    omega : float, default=0.35
        peak width parameter for broadening
    include_name : bool, default=True
        whether to include molecule name in output files
    exclude_mabe : bool, default=False
        whether to exclude Mean Absolute Binding Energy calculation
    gmp : float, default=0.003
        percentage of the maximum peak height to cut off the plot on the x-axis

    Attributes
    ----------
    start : Start
        instance of the Start object containing initial setup and configuration
    graph : bool
        whether to generate and display a graphical plot of the XPS spectrum
    intensity : float
        peak intensity scaling factor for the spectrum
    asym : bool
        whether to apply asymmetric peak broadening
    a : float
        asymmetry parameter for peak broadening
    b : float
        secondary asymmetry parameter for peak broadening
    gl_ratio : float
        Gaussian-Lorentzian mixing ratio for pseudo-Voigt broadening
    omega : float
        peak width parameter for broadening
    include_name : bool
        whether to include molecule name in output files
    exclude_mabe : bool
        whether to exclude Mean Absolute Binding Energy calculation
    gmp : float
        graphical margin parameter for plotting
    """

    def __init__(  # noqa: PLR0913
        self,
        start: Start,
        graph: bool,
        intensity: float = 1,
        asym: bool = False,
        a: float = 0.2,
        b: float = 0.0,
        gl_ratio: float = 0.5,
        omega: float = 0.35,
        include_name: bool = True,
        exclude_mabe: bool = False,
        gmp: float = 0.003,
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
        checks_utils.check_args(("constrained_atom", self.start.constr_atom))

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
        source_file = f"{element}_xps_{file_type}.txt"
        destination = self.start.run_loc / f"{element}_xps_{file_type}.txt"
        shutil.move(source_file, destination)

    def call_broaden(self, xps: list[float]) -> npt.NDArray[np.float64]:
        """
        Apply pseudo-Voigt peak broadening.

        Parameters
        ----------
        xps : list[float]
            deltaSCF energies

        Returns
        -------
        peaks : npt.NDArray[np.float64]
            broadened peaks
        """
        return broaden(
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

    def write_spectrum_to_file(
        self, peaks: npt.NDArray[np.float64], element: str, bin_width: float = 0.01
    ) -> None:
        """
        Write the spectrum to a text file.

        Parameters
        ----------
        peaks : npt.NDArray[np.float64]
            broadened peaks
        element : str
            element the binding energies were calculated for
        bin_width : float
            resolution of the spectral curve - lower values increase the resolution
        """
        data = []
        bin_val = 0.0
        for peak in peaks:
            data.append(f"{bin_val} {peak}\n")
            bin_val += bin_width

        with open(f"{element}_xps_spectrum.txt", "w") as spec:
            spec.writelines(data)

    def plot_xps(self, xps: list[float]) -> None:
        """
        Plot the XPS spectrum and save as pdf and png files.

        Parameters
        ----------
        xps : list[float]
            list of individual binding energies.
        """
        xps_spec = XPSSpectrum(
            self.gmp, self.start.run_loc, self.start.constr_atom, self.include_name
        )

        print("\nplotting spectrum and calculating MABE...")

        xps_spec.plot(xps, self.exclude_mabe)


class Projector(calculations_utils.GroundCalc, calculations_utils.ExcitedCalc):
    """
    Force occupation of Kohn-Sham states and project onto basis functions.

    ...

    Parameters
    ----------
    start : Start
        instance of the Start object
    run_type : Literal["ground", "init_1", "init_2", "hole"]
        type of calculation to perform
    occ_type : Literal["deltascf_projector", "force_occupation_projector"]
        use either the refactored or original projector keyword
    pbc : tuple[int, int, int]
        k-grid for a periodic calculation
    spin : Literal[1, 2]
        spin channel of the constraint
    ks_range : tuple[int, int]
        range of Kohn-Sham states to constrain
    control_opts : tuple[str, ...]
        additional control options to be added to the control.in file
    """

    def __init__(
        self,
        start: Start,
        run_type: Literal["ground", "init_1", "init_2", "hole"],
        occ_type: Literal["deltascf_projector", "force_occupation_projector"],
        pbc: tuple[int, int, int],
        spin: Literal[1, 2],
        ks_range: tuple[int, int],
        control_opts: tuple[str, ...],
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
        super(calculations_utils.GroundCalc, self).__init__(start)

        self.start = start
        self.run_type: Literal["ground", "init_1", "init_2", "hole"] = run_type
        self.occ_type: Literal["deltascf_projector", "force_occupation_projector"] = (
            occ_type
        )
        self.pbc = pbc
        self.spin: Literal[1, 2] = spin
        self.ks_range = ks_range

        self.ground_geom = self.start.run_loc / "ground/geometry.in"
        self.ground_control = self.start.run_loc / "ground/control.in"

        # Convert control options to a dictionary
        self.control_opts = control_utils.convert_opts_to_dict(control_opts, pbc)

        # Raise a warning if no additional control options have been specified
        calculations_utils.warn_no_extra_control_opts(
            self.control_opts, start.control_input
        )

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
        check_restart : bool, default=True
            Whether to check for the existance of restart files or not
        check_args : bool, default=False
            Whether to check if the required CLI arguments were given or not
        """
        # Check that the previous calculation has been run
        prev_calc = self.check_prereq_calc(
            current_calc, self.start.constr_atom, "projector"
        )

        # Check that the current calculation has not already been run
        checks_utils.check_curr_prev_run(
            self.run_type,
            self.start.run_loc,
            self.start.constr_atom,
            self.atom_specifier,
            "projector",
            self.start.hpc,
            self.start.force,
        )

        # Check that the restart files exist from the previous calculation
        if check_restart:
            for i_atom in self.atom_specifier:
                self.check_restart_files(self.start.constr_atom, prev_calc, i_atom)

        # Check that the constrained atoms have been given
        if current_calc != "hole":
            checks_utils.check_params(self.start)
        else:
            checks_utils.check_params(self.start, include_hpc=False)

        # Check required arguments have been given
        if check_args:
            checks_utils.check_args(("ks_range", self.ks_range))

    def _call_setups(self, proj: force_occupation.Projector) -> None:
        """
        Set up files and parameters for the initialisation and hole calculations.

        Parameters
        ----------
        proj : force_occupation.Projector
            Instance of Projector
        """
        proj.setup_init_1(
            self.ks_range,
            self.ground_control,
            checks_utils.check_lattice_vecs(self.start.geometry_input),
        )
        proj.setup_init_2(
            self.ks_range,
            self.start.occupation,
            self.occ_type,
            self.spin,
            checks_utils.check_lattice_vecs(self.start.geometry_input),
        )
        proj.setup_hole(
            self.ks_range,
            self.start.occupation,
            self.occ_type,
            self.spin,
            checks_utils.check_lattice_vecs(self.start.geometry_input),
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
        next(
            (self.start.run_loc / f"{self.start.constr_atom}{atom}" / begin).glob(
                "*restart*"
            )
        ).is_file()

        source_pattern = self.start.run_loc / f"{self.start.constr_atom}{atom}" / begin
        dest_dir = self.start.run_loc / f"{self.start.constr_atom}{atom}" / end

        # Copy all restart files from source to destination
        for restart_file in source_pattern.glob("*restart*"):
            shutil.copy2(restart_file, dest_dir)

    def check_periodic(self) -> None:
        """Check if the lattice vectors and k_grid have been provided."""
        print(
            "-p/--pbc argument not given, attempting to use"
            " k_grid from control file or previous calculation"
        )

        # Try to parse the k-grid if other calculations have been run
        try:
            pbc_list = []
            for control in self.start.run_loc.rglob("control.in"):
                with control.open() as control_lines:
                    for line in control_lines:
                        if "k_grid" in line:
                            pbc_list.extend([line.split()[1:]])

            # If different k_grids have been used for different calculations,
            # then enforce the user to provide the k_grid
            if pbc_list.count(pbc_list[0]) == len(pbc_list):
                pbc_list = tuple([int(i) for i in pbc_list[0]])
                self.control_opts["k_grid"] = pbc_list
            else:
                raise click.MissingParameter(param_hint="-p/--pbc", param_type="option")

        except IndexError as err:
            raise click.MissingParameter(
                param_hint="-p/--pbc", param_type="option"
            ) from err

    def setup_excited(self) -> tuple[list[int], str]:
        """
        Set up files and parameters required for the init and hole calculations.

        Returns
        -------
            spec_run_info : str
                redirection location for STDERR of calculation
        """
        # Get the atom indices to constrain
        self.atom_specifier = geometry_utils.get_atoms(
            self.ground_geom, self.start.constr_atom
        )

        self._calc_checks("init_1", check_restart=False, check_args=True)

        # Create the ForceOccupation object
        proj = force_occupation.Projector(
            self.start.constr_atom,
            self.start.run_loc,
            self.ground_geom,
            self.control_opts,
            self.atom_specifier,
            self.start.use_extra_basis,
        )

        # TODO allow this for multiple constrained atoms using n_atoms
        proj.get_electronic_structure(self.start.constr_atom)

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
        # Get the atom indices to constrain
        self.atom_specifier = geometry_utils.get_atoms(
            self.ground_geom, self.start.constr_atom
        )

        self._calc_checks("init_2")

        # Add any additional options to the control file
        for i in range(len(self.atom_specifier)):
            if len(self.control_opts) > 0:
                control_utils.add_control_opts(
                    self.start,
                    self.start.constr_atom,
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
        # Get the atom indices to constrain
        self.atom_specifier = geometry_utils.get_atoms(
            self.ground_geom, self.start.constr_atom
        )

        # Check for if init_2 hasn't been run
        if self.start.hpc:
            self._calc_checks("hole", check_restart=False, check_args=True)

            # Create the ForceOccupation object
            proj = force_occupation.Projector(
                self.start.constr_atom,
                self.start.run_loc,
                self.ground_geom,
                self.control_opts,
                self.atom_specifier,
            )

            proj.get_electronic_structure(self.start.constr_atom)

            # Setup files required for the initialisation and hole calculations
            self._call_setups(proj)
        else:
            self._calc_checks("hole")

        # Add a tag to the geometry file to identify the molecule name
        if self.start.spec_mol is not None:
            geometry_utils.add_molecule_identifier(self.start, self.atom_specifier)

        # Add any additional control options to the hole control file
        for i in [*list(range(*self.ks_range)), self.ks_range[-1]]:
            if len(self.control_opts) > 0:
                control_utils.add_control_opts(
                    self.start,
                    self.start.constr_atom,
                    i,
                    "hole",
                    self.control_opts,
                )

            # Copy the restart files to hole from init_2
            if not self.start.hpc:
                self._cp_restart_files(self.atom_specifier[i], "init_2", "hole")

        # Do not redirect STDERR to /dev/null as not converged errors should not occur
        # here
        spec_run_info = ""

        return self.atom_specifier, spec_run_info


class Basis(calculations_utils.GroundCalc, calculations_utils.ExcitedCalc):
    """
    Force occupation of the basis states directly.

    ...

    Parameters
    ----------
    start : Start
        Instance of the Start object
    run_type : Literal["ground", "hole"]
        Type of calculation to perform
    occ_type : Literal["deltascf_basis", "force_occupation_basis"]
        Method for constraining the occupation
    spin : Literal[1, 2]
        Spin channel of the constraint
    n_qn : int
        Principal quantum number for the basis function to constrain
    l_qn : int
        Angular momentum quantum number for the basis function to constrain
    m_qn : int
        Magnetic quantum number for the basis function to constrain
    ks_max : int
        Highest energy Kohn-Sham state to include in the MOM
    control_opts : tuple[str, ...]
        Additional control options to be added to the control.in file
    """

    def __init__(
        self,
        start: Start,
        run_type: Literal["ground", "hole"],
        occ_type: Literal["deltascf_basis", "force_occupation_basis"],
        spin: Literal[1, 2],
        n_qn: int,
        l_qn: int,
        m_qn: int,
        ks_max: int,
        control_opts: tuple[str, ...],
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
        super(calculations_utils.GroundCalc, self).__init__(start)

        self.start = start
        self.run_type: Literal["ground", "hole"] = run_type
        self.occ_type: Literal["deltascf_basis", "force_occupation_basis"] = occ_type
        self.spin: Literal[1, 2] = spin
        self.n_qn = n_qn
        self.l_qn = l_qn
        self.m_qn = m_qn
        self.ks_max = ks_max

        self.ground_geom = self.start.run_loc / "ground/geometry.in"
        self.ground_control = self.start.run_loc / "ground/control.in"

        # Convert control options to a dictionary
        self.control_opts = control_utils.convert_opts_to_dict(control_opts, None)

        # Raise a warning if no additional control options have been specified
        calculations_utils.warn_no_extra_control_opts(
            self.control_opts, start.control_input
        )

    def _calc_checks(self) -> None:
        """Perform checks before running the excited calculation."""
        # Check that the current calculation has not already been run
        checks_utils.check_curr_prev_run(
            self.run_type,
            self.start.run_loc,
            self.start.constr_atom,
            self.atom_specifier,
            "basis",
            self.start.hpc,
            self.start.force,
        )

        # Check that the constrained atoms have been given
        checks_utils.check_params(self.start, include_hpc=False)

        # Check that the required arguments have been given
        checks_utils.check_args(
            ("ks_max", self.ks_max),
            ("n_qn", self.n_qn),
            ("l_qn", self.l_qn),
            ("m_qn", self.m_qn),
        )

    def setup_excited(self) -> list[int]:
        """
        Set up files and parameters required for the hole calculation.

        Returns
        -------
        list[int]
            Indices for atoms as specified in geometry.in
        """
        # Do this outside of _calc_checks as atom_specifier is needed for that function
        self.check_prereq_calc("hole", self.start.constr_atom, "basis")

        # Get the atom indices to constrain
        self.atom_specifier = geometry_utils.get_atoms(
            self.ground_geom, self.start.constr_atom
        )

        self._calc_checks()

        # Create the directories required for the hole calculation
        basis = force_occupation.Basis(
            self.start.constr_atom,
            self.start.run_loc,
            self.ground_geom,
            self.control_opts,
            self.atom_specifier,
        )

        basis.setup_basis(
            self.spin,
            self.n_qn,
            self.l_qn,
            self.m_qn,
            self.start.occupation,
            self.ks_max,
            self.occ_type,
        )

        # Add molecule ID to geometry file
        if self.start.spec_mol is not None:
            geometry_utils.add_molecule_identifier(
                self.start, self.atom_specifier, basis=True
            )

        return self.atom_specifier
