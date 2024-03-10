import glob
import os
import warnings
from pathlib import Path
from typing import List, Literal, Tuple, Union

import click
import numpy as np
from ase import Atoms
from ase.io import read

import deltascf_aims.calc_dscf as cds
import dscf_utils.main_utils as du
from deltascf_aims.force_occupation import Basis, ForceOccupation, Projector
from deltascf_aims.plot import XPSSpectrum
from deltascf_aims.schmid_pseudo_voigt import broaden
from dscf_utils.main_utils import ExcitedCalc, GroundCalc


class Start(object):
    """
    Perform initial checks and setup for running calcultions.

    ...

    Attributes
    ----------
        hpc : bool
            setup a calculation primarily for use on a HPC cluster WITHOUT running the
            calculation
        geometry_input : click.File()
            specify a custom geometry.in instead of using a structure from PubChem or ASE
        control_input : click.File()
            specify a custom control.in instead of automatically generating one
        binary : bool
            modify the path to the FHI-aims binary
        run_location : click.Path(file_okay=False, dir_okay=True)
            optionally specify a custom location to run the calculation
        spec_mol : str
            molecule to be used in the calculation
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
        geometry_input,
        control_input,
        binary,
        run_location,
        spec_mol,
        constr_atom,
        spec_at_constr,
        occupation,
        n_atoms,
        basis_set,
        use_extra_basis,
        graph,
        print_output,
        nprocs,
    ):
        self.hpc = hpc
        self.geometry_input = geometry_input
        self.control_input = control_input
        self.binary = binary
        self.run_loc = run_location
        self.spec_mol = spec_mol
        self.constr_atom = constr_atom
        self.spec_at_constr = spec_at_constr
        self.occupation = occupation
        self.n_atoms = n_atoms
        self.basis_set = basis_set
        self.use_extra_basis = use_extra_basis
        self.graph = graph
        self.print_output = print_output
        self.nprocs = nprocs

        self.ase = True

    # def _check_help_arg(func):
    #     """
    #     Print click help if --help flag is given.
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
        """Check for lattice vectors and k-grid in input file"""

        self.found_lattice_vecs = False
        if self.geometry_input is not None:
            du.check_constrained_geom(self.geometry_input)
            self.found_l_vecs = du.check_lattice_vecs(self.geometry_input)
        else:
            self.found_l_vecs = False

        self.found_k_grid = False
        if self.control_input is not None:
            self.found_k_grid = du.check_k_grid(self.control_input)
        else:
            self.found_k_grid = False

    def check_ase_usage(self) -> None:
        """Check whether ASE should be used or not."""

        if self.control_input is not None:
            self.ase = False  # Do not use if control.in is specified

    # @_check_help_arg
    def create_structure(self) -> Union[Atoms, List[Atoms]]:
        """Initialise an ASE atoms object from geometry file if given or find databases
        if not.

        Returns
        -------
        Union[Atoms, List[Atoms]]
            ASE atoms object

        Raises
        ------
        click.MissingParameter
            The param_hint option has not been provided

        """

        atoms = Atoms()

        # Find the structure if not given
        if self.spec_mol is None and self.geometry_input is None:
            try:
                atoms = read(f"./{self.run_loc}/ground/geometry.in")
                print(
                    "molecule argument not provided, defaulting to using existing geometry.in"
                    " file"
                )
            except FileNotFoundError:
                raise click.MissingParameter(
                    param_hint="-m/--molecule or -e/--geometry_input",
                    param_type="option",
                )

        # Build the structure if given
        elif self.ase:
            if self.spec_mol is not None:
                atoms = du.build_geometry(self.spec_mol)
            if self.geometry_input is not None:
                atoms = read(self.geometry_input.name)

        return atoms

    def find_constr_atom_element(self, atoms) -> None:
        """Find the element of the atom to perform XPS/NEXAFS for.

        Parameters
        ----------
        atoms : Atoms
            Element of constr_atom

        """

        # TODO: support for multiple constrained atoms
        for atom in atoms:
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
        with open(f"{current_path}/aims_bin_loc.txt", "r") as f:
            lines = f.readlines()
            if "~" in lines[0][:-1]:
                raise FileNotFoundError(
                    "Please provide the full path to the FHI-aims binary"
                )
            try:
                bin_path = lines[0][:-1]
            except IndexError:
                bin_path = ""

        return current_path, bin_path

    def bin_path_prompt(self, current_path, bin_path) -> str:
        """
        Ensure the user has entered the path to the binary. If not open the user's
        $EDITOR to allow them to enter the path.

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
                with open(f"{current_path}/aims_bin_loc.txt", "w") as f:
                    f.write(bin_line)

                with open(f"{current_path}/aims_bin_loc.txt", "r") as f:
                    self.binary = f.readlines()[0]

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

    def check_species_path(self, binary) -> None:
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
            raise NotADirectoryError(
                f"species_defaults directory not found in {Path(binary).parent.parent}"
            )

    @property
    def atoms(self):
        """
        ASE atoms object

        Returns
        -------
            _atoms : Atoms
                ASE atoms object
        """

        return self._atoms

    @atoms.setter
    def atoms(self, atoms):
        """
        Set the ASE atoms object

        Parameters
        ----------
            atoms : Atoms
                ASE atoms object
        """

        self._atoms = atoms

    def add_calc(self, atoms, binary) -> Atoms:
        """
        Add an ASE calculator to an Atoms object.

        Parameters
        __________
            atoms : Atoms
                ASE atoms object
            binary : str
                path to the location of the FHI-aims binary

        Returns
        -------
            atoms : Atoms
                ASE atoms object with a calculator added
        """

        atoms.calc = du.create_calc(self.nprocs, binary, self.species, self.basis_set)

        if self.print_output:
            warnings.warn("-p/--print_output is not supported with the ASE backend")

        return atoms


class Process:
    """
    Calculate DSCF values and plot the simulated XPS spectra.

    ...

    Attributes
    ----------

    """

    def __init__(
        self,
        start,
        gmp,
        intensity=1,
        asym=False,
        a=0.2,
        b=0.0,
        gl_ratio=0.5,
        omega=0.35,
    ):
        self.start = start
        self.gmp = gmp
        self.intensity = intensity
        self.asym = asym
        self.a = a
        self.b = b
        self.gl_ratio = gl_ratio
        self.omega = omega

    def calc_dscf_energies(self) -> Tuple[List[float], str]:
        """
        Parse absolute energies and calculate deltaSCF energies

        Returns
        -------
            xps : List[float]
                deltaSCF energies
        """

        grenrgys = cds.read_ground(self.start.run_location)
        excienrgys, element = cds.read_atoms(
            self.start.run_location, self.start.constr_atom
        )
        xps = cds.calc_delta_scf(self.start.constr_atom, grenrgys, excienrgys)

        return xps, element

    def move_file_to_run_loc(self, element, type: Literal["peaks", "spectrum"]) -> None:
        """
        Move either the peaks or spectrum file to the run location.

        Parameters
        ----------
            element : str
                element the binding energies were calculated for
            type : Literal["peaks", "spectrum"]
                type of file to move
        """

        os.system(f"mv {element}_xps_{type}.txt {self.start.run_loc}")

    def call_broaden(self, xps) -> np.ndarray:
        """
        Apply pseudo-Voigt peak broadening.

        Parameters
        ----------
            xps : List[float]
                deltaSCF energies

        Returns
        -------
            peaks : np.ndarray
                broadened peaks
        """

        peaks, _ = broaden(
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
            data.append(f"{str(bin_val)} {str(peak)}\n")
            bin_val += bin_width

        with open(f"{element}_xps_spectrum.txt", "w") as spec:
            spec.writelines(data)

    def plot_xps(self, xps):
        """
        Plot the XPS spectrum and save as pdf and png files.

        Parameters
        ----------
            xps : list
                list of individual binding energies
        """
        xps_spec = XPSSpectrum(self.gmp, self.start.run_loc, self.start.constr_atom)

        print("\nplotting spectrum and calculating MABE...")

        xps_spec.plot(xps)


class ProjectorWrapper(GroundCalc, ExcitedCalc):
    """
    Force occupation of the basis functions by projecting the occupation of Kohn-Sham
    states onto them.

    ...

    Attributes
    ----------
        start : Start
            instance of the Start object
        run_type : click.Choice(["ground", "init_1", "init_2", "hole"])
            type of calculation to perform
        occ_type : click.Choice(["deltascf_projector", "force_occupation_projector"])
            use either the refactored or original projector keyword
        pbc : tuple
            k-grid for a periodic calculation
        l_vecs : List[List[float]]
            lattice vectors in a 3x3 matrix of floats
        spin : click.Choice(["1", "2"])
            spin channel of the constraint
        ks_range : click.IntRange(1)
            range of Kohn-Sham states to constrain
        control_opts : Tuple[str]
            additional control options to be added to the control.in file

    Methods
    -------
        check_periodic()
            Check if the lattice vectors and k_grid have been provided
        _call_setups(proj)
            Setup files and parameters required for the initialisation and hole
            calculations
        _check_params(include_hpc=True)
            Check that the parameters given in Start are valid
        _check_prev_runs(prev_calc, atom)
            Check if the required previous calculation has been run
        _cp_restart_files(atom, begin, end)
            Copy the restart files from one calculation location to another
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
        super(ProjectorWrapper, self).__init__(
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
        self.pbc = pbc
        self.l_vecs = l_vecs
        self.spin = spin
        self.ks_range = ks_range

        self.ground_geom = f"{self.start.run_loc}/ground/geometry.in"
        self.ground_control = f"{self.start.run_loc}/ground/control.in"

        # Convert control options to a dictionary
        self.control_opts = du.convert_opts_to_dict(control_opts, pbc)

        # Raise a warning if no additional control options have been specified
        du.warn_no_extra_control_opts(self.control_opts, start.control_input)

        # Convert constr_atom to a list
        if not isinstance(self.start.constr_atom, list):
            self.constr_atoms = [self.start.constr_atom]
        else:
            self.constr_atoms = self.start.constr_atom

    def _calc_checks(
        self, prev_calc, current_calc, check_restart=True, check_args=False
    ) -> None:
        """
        Perform checks before running an excited calculation.

        Parameters
        ----------
        prev_calc : str
            Previous calculation that should have been run
        current_calc : str
            Type of excited calculation that will be run
        check_restart : bool
            Whether to check for the existance of restart files or not
        check_args : bool
            Whether to check if the required CLI arguments were given or not
        """

        # Check that the current calculation has not already been run
        du.check_curr_prev_run(
            self.run_type,
            self.start.run_loc,
            self.constr_atoms,
            self.atom_specifier,
            "projector",
            self.start.hpc,
        )

        # Check that the restart files exist from the previous calculation
        if check_restart:
            for i_atom in self.atom_specifier:
                self.check_restart_files(self.constr_atoms, prev_calc, i_atom)

        # Check that the constrained atoms have been given
        if current_calc != "hole":
            du.check_params(self.start)
        else:
            du.check_params(self.start, include_hpc=False)

        # Check required arguments have been given
        if check_args:
            du.check_args(("ks_range", self.ks_range))

    def _call_setups(self, proj) -> None:
        """
        Setup files and parameters required for the initialisation and hole
        calculations.

        Parameters
        ----------
            proj : ForceOccupation
                Projector instance of ForceOccupation
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

    def _cp_restart_files(self, atom, begin, end) -> None:
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
                f"{self.start.run_loc}/{self.constr_atoms[0]}{atom}/{begin}/"
                "*restart*"
            )[0]
        )
        os.system(
            f"cp {self.start.run_loc}/{self.constr_atoms[0]}{atom}/{begin}/"
            f"*restart* {self.start.run_loc}/{self.constr_atoms[0]}{atom}"
            f"/{end}/"
        )

    def _get_atom_indices(self, fo) -> List[int]:
        """
        Get atom indices from the ground state geometry file

        Parameters
        ----------
            fo : object
                ForceOccupation object

        Returns
        -------
            atom_specifier : List[int]
                atom indices to constrain
        """
        atom_specifier = fo.get_atoms(
            self.constr_atoms, self.start.spec_at_constr, self.ground_geom
        )

        return atom_specifier

    def check_periodic(self) -> None:
        """
        Check if the lattice vectors and k_grid have been provided.
        """

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
                with open(control, "r") as control:
                    for line in control:
                        if "k_grid" in line:
                            pbc_list.append(line.split()[1:])

            # If different k_grids have been used for different calculations,
            # then enforce the user to provide the k_grid
            if not pbc_list.count(pbc_list[0]) == len(pbc_list):
                raise click.MissingParameter(param_hint="-p/--pbc", param_type="option")
            else:
                pbc_list = tuple([int(i) for i in pbc_list[0]])
                self.control_opts["k_grid"] = pbc_list

        except IndexError:
            raise click.MissingParameter(param_hint="-p/--pbc", param_type="option")

    def setup_excited(self) -> Tuple[List[int], str]:
        """
        Setup files and parameters required for the init and hole calculations.

        Returns
        -------
            spec_run_info : str
                redirection location for STDERR of calculation
        """

        # Get element symbols to constrain
        element_symbols = du.get_element_symbols(
            self.ground_geom, self.start.spec_at_constr
        )

        # Check that the prerequisite calculation has been run
        # This has to be done outside of _calc_checks as that function requires
        # atom_specifier which is set in a function that requires ForceOccupation to be
        # initialised.
        prev_calc = self.check_prereq_calc("init_1", self.constr_atoms, "projector")

        # Create the ForceOccupation object
        fo = ForceOccupation(
            element_symbols,
            self.start.run_loc,
            self.ground_geom,
            self.control_opts,
            f"{self.start.species}/defaults_2020/{self.start.basis_set}",
            self.start.use_extra_basis,
        )

        # Get the atom indices to constrain
        self.atom_specifier = self._get_atom_indices(fo)

        self._calc_checks(prev_calc, "init_1", check_restart=False, check_args=True)

        # TODO allow this for multiple constrained atoms using n_atoms
        for atom in element_symbols:
            fo.get_electronic_structure(atom)

        proj = Projector(fo)
        self._call_setups(proj)

        spec_run_info = ""

        return self.atom_specifier, spec_run_info

    def pre_init_2(self) -> Tuple[List[int], str]:
        """
        Prerequisite before running the 2nd init calculation.

        Returns
        -------
            spec_run_info : str
                Redirection location for STDERR of calculation
        """

        # Get element symbols to constrain
        element_symbols = du.get_element_symbols(
            self.ground_geom, self.start.spec_at_constr
        )

        # Check that the prerequisite calculation has been run
        # This has to be done outside of _calc_checks as that function requires
        # atom_specifier which is set in a function that requires ForceOccupation to be
        # initialised.
        prev_calc = self.check_prereq_calc("init_2", self.constr_atoms, "projector")

        # Create the ForceOccupation object
        fo = ForceOccupation(
            element_symbols,
            self.start.run_loc,
            self.ground_geom,
            self.control_opts,
            f"{self.start.species}/defaults_2020/{self.start.basis_set}",
            self.start.use_extra_basis,
        )

        # Get the atom indices to constrain
        self.atom_specifier = self._get_atom_indices(fo)

        # Add any additional options to the control file
        for i in range(len(self.atom_specifier)):
            self._calc_checks(prev_calc, "init_2")

            if len(self.control_opts) > 0 or self.start.control_input:
                du.add_control_opts(
                    self.start,
                    self.constr_atoms,
                    self.control_opts,
                    self.atom_specifier[i],
                    "init_2",
                )

            # Copy the restart files to init_2 from init_1
            self._cp_restart_files(self.atom_specifier[i], "init_1", "init_2")

        # Prevent SCF not converged errors from printing
        # It could be an issue to do this if any other errors occur
        spec_run_info = " 2>/dev/null"

        return self.atom_specifier, spec_run_info

    def pre_hole(self) -> Tuple[List[int], str]:
        """
        Prerequisite to running the hole calculation

        Returns
        -------
        Tuple[List[int], str]
            Indices for atoms as specified in geometry.in, redirection location for
            STDERR of calculation

        """

        # Get element symbols to constrain
        element_symbols = du.get_element_symbols(
            self.ground_geom, self.start.spec_at_constr
        )

        # Check that the prerequisite calculation has been run
        # This has to be done outside of _calc_checks as that function requires
        # atom_specifier which is set in a function that requires ForceOccupation to be
        # initialised.
        prev_calc = self.check_prereq_calc("hole", self.constr_atoms, "projector")

        # Create the ForceOccupation object
        fo = ForceOccupation(
            element_symbols,
            self.start.run_loc,
            self.ground_geom,
            self.control_opts,
            f"{self.start.species}/defaults_2020/{self.start.basis_set}",
            self.start.use_extra_basis,
        )

        # Get the atom indices to constrain
        self.atom_specifier = self._get_atom_indices(fo)

        if self.start.hpc:
            self._calc_checks(None, "hole", check_restart=False, check_args=True)

            for atom in element_symbols:
                fo.get_electronic_structure(atom)

            # Setup files required for the initialisation and hole calculations
            proj = Projector(fo)
            self._call_setups(proj)

        # Add a tag to the geometry file to identify the molecule name
        if self.start.spec_mol is not None:
            du.add_molecule_identifier(self.start, self.atom_specifier)

        # Add any additional control options to the hole control file
        for i in range(len(self.atom_specifier)):

            # Check for if init_2 hasn't been run
            if not self.start.hpc:
                self._calc_checks(prev_calc, "hole")

            if len(self.control_opts) > 0 or self.start.control_input:
                du.add_control_opts(
                    self.start,
                    self.constr_atoms,
                    self.control_opts,
                    self.atom_specifier[i],
                    "hole",
                )

            # Copy the restart files to hole from init_2
            if not self.start.hpc:
                self._cp_restart_files(self.atom_specifier[i], "init_2", "hole")

        # Don't redirect STDERR to /dev/null as not converged errors should not occur
        # here
        spec_run_info = ""

        return self.atom_specifier, spec_run_info


class BasisWrapper(GroundCalc, ExcitedCalc):
    """
    Force occupation of the basis states directly.

    ...

    Attributes
    ----------
        start : Start
            instance of the Start object
        run_type : click.Choice(["ground", "hole"])
            type of calculation to perform
        atom_index : click.IntRange(1)
            atom index to constrain
        occ_type : click.Choice(["deltascf_projector", "force_occupation_projector"])
            use either the refactored or original projector keyword
        multiplicity : click.IntRange(1)
            multiplicity of the system
        n_qn : click.IntRange(1)
            principal quantum number
        l_qn : click.IntRange(0)
            angular momentum quantum number
        m_qn : click.IntRange(-l_qn, l_qn)
            magnetic quantum number
        ks_max : click.IntRange(1)
            maximum Kohn-Sham state to constrain
        control_opts : Tuple[str]
            additional control options to be added to the control.in file
    """

    def __init__(
        self,
        start,
        run_type,
        atom_index,
        occ_type,
        multiplicity,
        n_qn,
        l_qn,
        m_qn,
        ks_max,
        control_opts,
    ):
        super(BasisWrapper, self).__init__(
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
        self.atom_index = atom_index
        self.occ_type = occ_type
        self.multiplicity = multiplicity
        self.n_qn = n_qn
        self.l_qn = l_qn
        self.m_qn = m_qn
        self.ks_max = ks_max
        self.control_opts = control_opts

        self.ground_geom = f"{self.start.run_loc}/ground/geometry.in"
        self.ground_control = f"{self.start.run_loc}/ground/control.in"

        # Convert control options to a dictionary
        control_opts = du.convert_opts_to_dict(control_opts, None)

        # Raise a warning if no additional control options have been specified
        du.warn_no_extra_control_opts(self.control_opts, start.control_input)

        # Convert constr_atoms to a list
        if not isinstance(self.start.constr_atom, list):
            self.constr_atoms = [self.start.constr_atom]
        else:
            self.constr_atoms = self.start.constr_atom

    # def _add_control_keywords(self, fo, atom_specifier) -> None:
    #     """
    #     Add additional options to the control file.

    #     Parameters
    #     ----------
    #         fo : object
    #             ForceOccupation object
    #         atom_specifier : List[int]
    #             atom indices to constrain
    #     """

    #     # TODO allow multiple constraints using n_atoms
    #     for i in range(len(atom_specifier)):
    #         i += 1

    #         if len(self.control_opts) > 0 or self.start.control:
    #             du.add_control_opts(self.start, fo, self.control_opts, i, "hole")

    # def _add_geometry_tag(self, fo, constr_atoms, ground_geom) -> None:
    #     """
    #     Add the name of the molecule to the geometry file.

    #     Parameters
    #     ----------
    #         fo : object
    #             ForceOccupation object
    #         constr_atoms : List[str]
    #             atom indices to constrain
    #         ground_geom : str
    #             path to the ground state geometry file
    #     """

    #     # Get the atom indices from the ground state geometry file
    #     atom_specifier = fo.get_atoms(
    #         constr_atoms, self.start.spec_at_constr, ground_geom
    #     )

    #     # Add the tag
    #     du.add_molecule_identifier(self.start, atom_specifier)

    def _calc_checks(self) -> None:
        """
        Perform checks before running the excited calculation.
        """

        # Check that the current calculation has not already been run
        du.check_curr_prev_run(
            self.run_type,
            self.start.run_loc,
            self.constr_atoms,
            self.atom_specifier,
            "basis",
            self.start.hpc,
        )

        # Check that the constrained atoms have been given
        du.check_params(self.start)

        # Check that the required arguments have been given
        du.check_args(
            ("atom_index", self.atom_index),
            ("ks_max", self.ks_max),
            ("n_qn", self.n_qn),
            ("l_qn", self.l_qn),
            ("m_qn", self.m_qn),
        )

    def setup_excited(self) -> Tuple[object, List[str], str]:
        """
        Setup files and parameters required for the hole calculation.

        Returns
        -------
            fo : object
                ForceOccupation object
            constr_atoms : List[str]
                atom indices to constrain
            ground_geom : str
                path to the ground state geometry file
        """

        # Get the element symbols to constrain
        element_symbols = du.get_element_symbols(self.geom, self.start.spec_at_constr)

        prev_calc = self.check_prereq_calc("hole", self.constr_atoms, "basis")

        self._calc_checks()

        # Create the directories required for the hole calculation
        fo = ForceOccupation(
            element_symbols,
            self.start.run_loc,
            ground_geom,
            self.control_opts,
            f"{self.start.species}/defaults_2020/{self.start.basis_set}",
        )

        basis = Basis(fo)
        basis.setup_basis(
            self.multiplicity,
            self.n_qn,
            self.l_qn,
            self.m_qn,
            self.start.occ,
            self.ks_max,
            self.occ_type,
            self.start.basis_set,
            self.start.species,
        )

        # Add molecule ID to geometry file
        self._add_geometry_tag(fo, constr_atoms, ground_geom)

        # Add any additional options to the control file
        self._add_control_keywords(fo, constr_atoms)

        return fo, constr_atoms, ground_geom
