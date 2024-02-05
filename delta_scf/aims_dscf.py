import glob
import os
import sys
import warnings
from pathlib import Path
from typing import List, Literal, Tuple, Union

import click
import numpy as np
from ase import Atoms
from ase.io import read

import delta_scf.calc_dscf as cds
import dscf_utils.main_utils as du
from delta_scf.force_occupation import Basis, ForceOccupation, Projector
from delta_scf.plot import XPSSpectrum
from delta_scf.schmid_pseudo_voigt import broaden
from dscf_utils.main_utils import GroundCalc


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
        graph : bool
            print the simulated XPS spectrum
        print_output : bool
            print the live output of the calculation
        nprocs : int
            number of processors to use

    Methods
    -------
        check_for_help_arg()
            Print click help if --help flag is given
        check_for_geometry()
            Check a geometry file exists if specific atom indices are to be constrained
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
        graph,
        print_output,
        nprocs,
    ):
        """
        Parameters
        ----------
            hpc : bool
                setup a calculation primarily for use on a HPC cluster WITHOUT running
                the calculation
            geometry_input : click.File
                specify a custom geometry.in instead of using a structure from PubChem
                or ASE
            control_input : click.File
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
            graph : bool
                print the simulated XPS spectrum
            print_output : bool
                print the live output of the calculation
            nprocs : int
                number of processors to use
        """
        self.hpc = hpc
        self.geometry_input = geometry_input
        self.control_input = control_input
        self.binary = binary
        self.run_location = run_location
        self.spec_mol = spec_mol
        self.constr_atom = constr_atom
        self.spec_at_constr = spec_at_constr
        self.occupation = occupation
        self.n_atoms = n_atoms
        self.basis_set = basis_set
        self.graph = graph
        self.print_output = print_output
        self.nprocs = nprocs

        # Pass global options to subcommands
        # ctx.ensure_object(dict)

    @staticmethod
    def check_for_help_arg() -> None:
        """
        Print click help if --help flag is given.
        """

        if "--help" in sys.argv:
            click.help_option()

    def check_for_geometry(self) -> None:
        """
        Check a geometry file exists if specific atom indices are to be constrained.
        """

        if len(self.spec_at_constr) > 0:
            if not self.geometry_input:
                raise click.MissingParameter(
                    param_hint="-e/--geometry_input", param_type="option"
                )

    def check_for_pbcs(self) -> None:
        """
        Check for lattice vectors and k_grid in input files.
        """

        found_lattice_vecs = False
        if self.geometry_input is not None:
            found_lattice_vecs = du.check_constrained_geom(self.geometry_input)
        else:
            self.geometry_input = None
            self.lattice_vecs = None

        found_k_grid = False
        if self.control_input is not None:
            found_k_grid = du.check_control_k_grid(self.control_input)
        else:
            self.control_input = None

        if found_lattice_vecs or found_k_grid:
            self.lattice_vecs = True
        else:
            self.lattice_vecs = False

    def check_ase_usage(self) -> None:
        """
        Check whether ASE should be used or not.
        """

        self.ase = True
        if self.control_input is not None:
            self.ase = False  # Do not use if control.in is specified

    def create_structure(self) -> Union[Atoms, List[Atoms], None]:
        """
        Initialise an ASE atoms object from geometry file if given or find from
        databases if not.

        Returns
        -------
            atoms : Atoms
                ASE atoms object
        """

        atoms = Atoms()

        # Find the structure if not given
        if self.spec_mol is None and self.geometry_input is None:
            try:
                atoms = read(f"./{self.run_location}/ground/geometry.in")
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
        """
        Find the element of the atom to perform XPS/NEXAFS for.

        Parameters
        ----------
            atoms : Atoms
                ASE atoms object

        Returns
        -------
            constr_atom : str
                element of constr_atom
        """

        # TODO: support for multiple constrained atoms
        if self.constr_atom is None:
            for atom in atoms:
                if atom.index in self.spec_at_constr:
                    self.constr_atom = atom.symbol
                    break

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
            try:
                bin_path = f.readlines()[0][:-1]
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
                "# Ensure that the binary is located in the build directory of FHIaims"
            )
            bin_line = click.edit(marker)
            if bin_line is not None:
                with open(f"{current_path}/aims_bin_loc.txt", "w") as f:
                    f.write(bin_line)

                with open(f"{current_path}/aims_bin_loc.txt", "r") as f:
                    binary = f.readlines()[0]

            else:
                raise FileNotFoundError(
                    "path to the FHI-aims binary could not be found"
                )

        elif Path(bin_path).exists():
            print(f"specified binary path: {bin_path}")
            binary = bin_path

        else:
            raise FileNotFoundError("path to the FHI-aims binary could not be found")

        return binary

    def check_species_path(self, binary) -> None:
        """
        Check if the species_defaults directory exists in the correct location.

        Parameters
        ----------
            binary : str
                path to the location of the FHI-aims binary
        """

        self.species = f"{Path(self.binary).parent.parent}/species_defaults/"

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

    def create_calculator(self, atoms, binary, basis_set) -> None:
        """
        Create an ASE calculator for the ground calculation.

        Parameters
        __________
            atoms : Atoms
                ASE atoms object
            binary : str
                path to the location of the FHI-aims binary
            basis_set : str
                basis set to use for the calculation
        """

        aims_calc = du.create_calc(self.nprocs, binary, self.species, basis_set)
        atoms.calc = aims_calc
        # self.ctx.obj["CALC"] = aims_calc

        if self.print_output:
            warnings.warn("-p/--print_output is not supported with the ASE backend")

    # def _export_context(
    #     self,
    #     atoms,
    #     binary,
    #     run_location,
    # ):
    #     # User specified context objects
    #     self.ctx.obj["ATOMS"] = atoms
    #     self.ctx.obj["SPEC_MOL"] = self.spec_mol
    #     self.ctx.obj["BINARY"] = binary
    #     self.ctx.obj["RUN_LOC"] = run_location
    #     self.ctx.obj["CONSTR_ATOM"] = self.constr_atom
    #     self.ctx.obj["SPEC_AT_CONSTR"] = self.spec_at_constr
    #     self.ctx.obj["OCC"] = self.occupation
    #     self.ctx.obj["N_ATOMS"] = n_atoms  # TODO
    #     self.ctx.obj["BASIS_SET"] = basis_set
    #     self.ctx.obj["GRAPH"] = graph
    #     self.ctx.obj["PRINT"] = print_output
    #     self.ctx.obj["NPROCS"] = nprocs
    #     self.ctx.obj["DEBUG"] = debug
    #     self.ctx.obj["HPC"] = hpc

    #     # Context objects created in main()
    #     self.ctx.obj["SPECIES"] = species
    #     self.ctx.obj["ASE"] = ase


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
        """
        Parameters
        ----------
            start : Start
                instance of the Start object
            intensity : float
                intensity of the peaks
            asym : bool
                use an asymmetric pseudo-Voigt peak shape
            a : float
                parameter to control the asymmetry of the pseudo-Voigt peak shape
            b : float
                parameter to control the asymmetry of the pseudo-Voigt peak shape
            gl_ratio : float
                ratio of Gaussian to Lorentzian broadening
            omega : float
                width of the Lorentzian broadening
            gmp : float
                Gaussian mixing parameter
        """

        self.start = start
        self.gmp = gmp
        self.intensity = intensity
        self.asym = asym
        self.a = a
        self.b = b
        self.gl_ratio = gl_ratio
        self.omega = omega

    def calc_dscf_energies(self) -> List[float]:
        """
        Parse absolute energies and calculate deltaSCF energies

        Returns
        -------
            xps : List[float]
                deltaSCF energies
        """

        grenrgys = cds.read_ground(self.start.run_location)
        excienrgys = cds.read_atoms(
            self.start.run_location, self.start.constr_atom, cds.contains_number
        )
        xps = cds.calc_delta_scf(self.start.constr_atom, grenrgys, excienrgys)

        return xps

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


class ProjectorWrapper(GroundCalc):
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
        _add_control_opts(fo, atom, calc)
            Add any additional control options to the control file
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
        """
        Parameters
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
        """

        super().__init__(
            start.run_loc,
            start.atoms,
            start.basis_set,
            start.species,
            start.ase,
            start.hpc,
        )

        self.start = start
        self.run_type = run_type
        self.occ_type = occ_type
        self.pbc = pbc
        self.l_vecs = l_vecs
        self.spin = spin
        self.ks_range = ks_range
        self.control_opts = control_opts

        # Raise a warning if no additional control options have been specified
        du.warn_no_extra_control_opts(self.control_opts, start.control_input)

        # Convert control options to a dictionary
        control_opts = du.convert_opts_to_dict(control_opts, pbc)

    def _calc_checks(self) -> None:
        """
        Perform checks before running an excited calculation.
        """

        # Check that the ground state calculation has been run
        du.check_ground_calc(self.start)

        # Check that the constrained atoms have been given
        du.check_params(self.start)

        # Check required arguments have been given
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
            self.start.occ,
            self.occ_type,
            self.spin,
            self.start.found_lattice_vecs,
        )
        proj.setup_hole(
            self.ks_range[0],
            self.ks_range[1],
            self.start.occ,
            self.occ_type,
            self.spin,
            self.start.found_lattice_vecs,
        )

    def _check_periodic(self) -> None:
        """
        Check if the lattice vectors and k_grid have been provided.
        """

        if self.start.found_lattice_vecs or self.l_vecs is not None:
            if self.pbc is None:
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
                        raise click.MissingParameter(
                            param_hint="-p/--pbc", param_type="option"
                        )
                    else:
                        pbc_list = tuple([int(i) for i in pbc_list[0]])
                        self.control_opts["k_grid"] = pbc_list

                except IndexError:
                    raise click.MissingParameter(
                        param_hint="-p/--pbc", param_type="option"
                    )

    def _check_prev_runs(self, prev_calc, atom) -> None:
        """
        Check if the required previous calculation has been run.

        Parameters
        ----------
            prev_calc : str
                name of the previous calculation to check
            atom : int
                atom to check for
        """

        if (
            len(
                glob.glob(
                    f"{self.start.run_loc}/{self.start.constr_atoms[0]}"
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
                f"{self.start.run_loc}/{self.start.constr_atoms[0]}{atom}/{begin}/"
                "*reself.start*"
            )[0]
        )
        os.system(
            f"cp {self.start.run_loc}/{self.start.constr_atoms[0]}{atom}/{begin}/"
            f"*reself.start* {self.start.run_loc}/{self.start.constr_atoms[0]}{atom}"
            f"/{end}/"
        )

    # TODO: remove and put outside of class in ifmain
    # def run_ground(self) -> None:
    #     """
    #     Run the ground state calculation.
    #     """

    #     self.run(
    #         self.start.geom_inp,
    #         self.start.control_inp,
    #         self.start.constr_atom,
    #         self.start.calc,
    #         self.control_opts,
    #         self.l_vecs,
    #         self.start.print_output,
    #         self.start.nprocs,
    #         self.start.binary,
    #     )

    def setup_excited_calculations(self) -> Tuple[object, List[int]]:
        """
        Setup files and parameters required for the initialisation and hole
        calculations.

        Returns
        -------
            fo : object
                ForceOccupation object
            atom_specifier : List[int]
                atom indices to constrain
        """

        self._calc_checks()

        (
            ground_geom,
            self.ground_control,
            constr_atoms,
            element_symbols,
        ) = du.prepare_excited_calcs(self)

        fo = ForceOccupation(
            element_symbols,
            self.start.run_loc,
            ground_geom,
            self.control_opts,
            f"{self.start.species}/defaults_2020/{self.start.basis_set}",
        )

        # Get atom indices from the ground state geometry file
        atom_specifier = fo.read_ground_inp(
            constr_atoms, self.start.spec_at_constr, ground_geom
        )

        # TODO allow this for multiple constrained atoms using n_atoms
        for atom in element_symbols:
            fo.get_electronic_structure(atom)

        proj = Projector(fo)
        self._call_setups(proj)

        return fo, atom_specifier

    def pre_init_2(self, fo, atom_specifier) -> str:
        """
        Prerequisite before running the 2nd init calculation.

        Parameters
        ----------
            fo : object
                ForceOccupation object
            atom_specifier : List[int]
                atom indices to constrain

        Returns
        -------
            spec_run_info : str
                Redirection location for STDERR of calculation
        """

        du.check_params(self.start)

        # Add any additional options to the control file
        for i in range(len(atom_specifier)):
            i += 1

            # Check that init_1 has been run
            self._check_prev_runs("init_1", i)

            if len(self.control_opts) > 0 or self.start.control_inp:
                du.add_control_opts(self.start, fo, self.control_opts, i, "init_2")

            # Copy the restart files to init_2 from init_1
            self._cp_restart_files(i, "init_1", "init_2")

        # Prevent SCF not converged errors from printing
        # It could be an issue to do this if any other errors occur
        spec_run_info = " 2>/dev/null"

        return spec_run_info

    def pre_hole(self, fo, atom_specifier) -> str:
        """
        Prerequisite before running the hole calculation

        Parameters
        ----------
            fo : object
                ForceOccupation object
            atom_specifier : List[int]
                atom indices to constrain

        Returns
        -------
            spec_run_info : str
                Redirection location for STDERR of calculation
        """

        du.check_params(self.start, include_hpc=False)

        if self.start.hpc:
            du.check_args(("ks_range", self.ks_range))

            for atom in self.start.element_symbols:
                fo.get_electronic_structure(atom)

            # Setup files required for the initialisation and hole calculations
            # proj = Projector(fo)
            # self._call_setups(proj)

        # Add a tag to the geometry file to identify the molecule name
        du.add_molecule_identifier(self.start, atom_specifier)

        if not self.start.hpc:
            # Add any additional control options to the hole control file
            for i in range(len(atom_specifier)):
                i += 1

                # Check for if init_2 hasn't been run
                self._check_prev_runs("init_2", i)

                if len(self.control_opts) > 0 or self.start.control_inp:
                    du.add_control_opts(self.start, fo, self.control_opts, i, "hole")

                # Copy the restart files to hole from init_2
                self._cp_restart_files(i, "init_2", "hole")

        # Don't redirect STDERR to /dev/null as not converged errors should not occur
        # here
        spec_run_info = ""

        return spec_run_info


class BasisWrapper(GroundCalc):
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
        """
        Parameters
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

        super().__init__(
            start.run_loc,
            start.atoms,
            start.basis_set,
            start.species,
            start.ase,
            start.hpc,
        )

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

        # Raise a warning if no additional control options have been specified
        du.warn_no_extra_control_opts(self.control_opts, start.control_input)

        # Convert control options to a dictionary
        control_opts = du.convert_opts_to_dict(control_opts, None)

    # TODO: remove and put outside of class in ifmain
    # def run_ground(self) -> None:
    #     """
    #     Run the ground state calculation.
    #     """

    #     self.run(
    #         self.start.geom_inp,
    #         self.start.control_inp,
    #         self.start.constr_atom,
    #         self.start.calc,
    #         self.control_opts,
    #         self.l_vecs,
    #         self.start.print_output,
    #         self.start.nprocs,
    #         self.start.binary,
    #     )

    def _add_control_keywords(self, fo, atom_specifier) -> None:
        """
        Add additional options to the control file.

        Parameters
        ----------
            fo : object
                ForceOccupation object
            atom_specifier : List[int]
                atom indices to constrain
        """

        # TODO allow multiple constraints using n_atoms
        for i in range(len(atom_specifier)):
            i += 1

            if len(self.control_opts) > 0 or self.start.control:
                du.add_control_opts(self.start, fo, self.control_opts, i, "hole")

    def _add_geometry_tag(self, fo, constr_atoms, ground_geom) -> None:
        """
        Add the name of the molecule to the geometry file.

        Parameters
        ----------
            fo : object
                ForceOccupation object
            constr_atoms : List[str]
                atom indices to constrain
            ground_geom : str
                path to the ground state geometry file
        """

        # Get the atom indices from the ground state geometry file
        atom_specifier = fo.read_ground_inp(
            constr_atoms, self.start.spec_at_constr, ground_geom
        )

        # Add the tag
        du.add_molecule_identifier(self.start, atom_specifier)

    def _pre_calc_checks(self) -> None:
        """
        Perform checks before running the excited calculation.
        """

        # Check that the ground state calculation has been run
        du.check_ground_calc(self.start)

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

    def setup_excited_calculation(self) -> Tuple[object, List[str], str]:
        """
        Setup files and parameters required for the hole calculations.

        Returns
        -------
            fo : object
                ForceOccupation object
            constr_atoms : List[str]
                atom indices to constrain
            ground_geom : str
                path to the ground state geometry file
        """

        ground_geom, _, constr_atoms, element_symbols = du.prepare_excited_calcs(self)

        self._pre_calc_checks()

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
