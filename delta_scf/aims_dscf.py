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

import dscf_utils.main_utils as du
from delta_scf.calc_dscf import CalcDeltaSCF as cds
from delta_scf.force_occupation import Basis, ForceOccupation, Projector
from delta_scf.plot import XPSSpectrum
from delta_scf.schmid_pseudo_voigt import broaden


class Start(object):
    """
    Point of controlling the program flow.

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
            geometry_input : click.File()
                specify a custom geometry.in instead of using a structure from PubChem
                or ASE
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
            found_lattice_vecs = du.check_geom(self.geometry_input)
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

    def check_ase_usage(self) -> bool:
        """
        Check whether ASE should be used or not.

        Returns
        -------
            ase : bool
                use ASE to build the structure
        """

        ase = True
        if self.control_input is not None:
            ase = False  # Do not use if control.in is specified

        return ase

    def create_structure(self, ase) -> Union[Atoms, List[Atoms], None]:
        """
        Initialise an ASE atoms object from geometry file if given or find from
        databases if not.

        Parameters
        ----------
            ase : bool
                use ASE to build the structure

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
        elif ase:
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

        species = f"{Path(self.binary).parent.parent}/species_defaults/"

        # TODO: check if the warnings module could be used here
        # Check if the species_defaults directory exists in the correct location
        if not Path(species).exists():
            print(
                "\nError: ensure the FHI-aims binary is in the 'build' directory of the FHI-aims"
                " source code directory, and that the 'species_defaults' directory exists"
            )
            raise NotADirectoryError(
                f"species_defaults directory not found in {Path(binary).parent.parent}"
            )

    def create_calculator(self, atoms, binary, species, basis_set) -> None:
        """
        Create an ASE calculator for the ground calculation.

        Parameters
        __________
            atoms : Atoms
                ASE atoms object
            binary : str
                path to the location of the FHI-aims binary
            species : str
                path to the species_defaults directory
            basis_set : str
                basis set to use for the calculation
        """

        aims_calc = du.create_calc(self.nprocs, binary, species, basis_set)
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
        intensity=1,
        asym=False,
        a=0.2,
        b=0.0,
        gl_ratio=0.5,
        omega=0.35,
        gmp=0.003,
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
        self.intensity = intensity
        self.asym = asym
        self.a = a
        self.b = b
        self.gl_ratio = gl_ratio
        self.omega = omega
        self.gmp = gmp

    def calc_dscf_energies(self) -> List[float]:
        grenrgys = cds.read_ground(self.start.run_location)
        element, excienrgys = cds.read_atoms(
            self.start.run_location, self.start.constr_atom, cds.contains_number
        )
        xps = cds.calc_delta_scf(element, grenrgys, excienrgys)

        return xps

    def move_file(self, element, type: Literal["peaks", "spectrum"]) -> None:
        os.system(f"mv {element}_xps_{type}.txt {self.start.run_loc}")

    def broaden(self, xps) -> np.ndarray:
        """
        Apply pseudo-Voigt peak broadening.
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

    def write_spectrum_to_file(self, peaks, element, bin_width=0.01):
        """
        Write the spectrum to a text file.
        """

        data = []
        bin_val = 0.00
        for peak in peaks:
            data.append(f"{str(bin_val)} {str(peak)}\n")
            bin_val += bin_width

        with open(f"{element}_xps_spectrum.txt", "w") as spec:
            spec.writelines(data)

    def plot_xps(self, start):
        xps_spec = XPSSpectrum(start.run_loc, start.constr_atom)

        print("\nplotting spectrum and calculating MABE...")
        Plot.sim_xps_spectrum(
            xps, ctx.obj["RUN_LOC"], ctx.obj["CONSTR_ATOM"], ctx.obj["AT_SPEC"][0], gmp
        )


class ProjectorWrapper:
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

        # Define class variables from args
        self.start = start
        self.run_type = run_type
        self.occ_type = occ_type
        self.pbc = pbc
        self.l_vecs = l_vecs
        self.spin = spin
        self.ks_range = ks_range
        self.control_opts = control_opts

        # Raise a warning if no additional control options have been specified
        if len(self.control_opts) < 1 and self.start.control_inp is None:
            warnings.warn(
                "No control options provided, using default options "
                "which can be found in the 'control.in' file"
            )

        # Convert control options to a dictionary
        control_opts = du.convert_opts_to_dict(control_opts, pbc)

    def check_periodic(self) -> None:
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

    def _call_setups(self, proj) -> None:
        """
        Setup files and parameters required for the initialisation and hole
        calculations.

        Parameters
        ----------
            proj : object
                Projector object
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

    def _check_params(self, include_hpc=True) -> None:
        """
        Check that the parameters given in Start are valid.

        Parameters
        ----------
            include_hpc : bool
                include the hpc parameter in the check
        """

        if include_hpc:
            if self.start.hpc:
                raise click.BadParameter(
                    "the -h/--hpc flag is only supported for the 'hole' run type"
                )

        if len(self.start.spec_at_constr) == 0 and len(self.start.constr_atom) == 0:
            raise click.BadParameter(
                "no atoms have been specified to constrain, please use "
                "-c/--constr_atoms or -s/--spec_at_constr options"
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
                    f"{atom}/{prev_calc}/*reself.start*"
                )
            )
            < 1
        ):
            print(
                f'{prev_calc} restart files not found, please ensure "{prev_calc}"'
                "has been run"
            )
            raise FileNotFoundError

    def _add_control_opts(self, fo, atom, calc) -> None:
        """
        Add any additional control options to the control file.

        Parameters
        ----------
            fo : object
                ForceOccupation object
            atom : int
                atom to add the control options to
            calc : str
                name of the calculation to add the control options to
        """

        parsed_control_opts = fo.get_control_keywords(
            f"{self.start.run_loc}/{self.start.constr_atoms[0]}{atom}/{calc}/control.in"
        )
        mod_control_opts = fo.mod_keywords(self.control_opts, parsed_control_opts)
        control_content = fo.change_control_keywords(
            f"{self.start.run_loc}/{self.start.constr_atoms[0]}{atom}/{calc}/control.in",
            mod_control_opts,
        )

        with open(
            f"{self.start.run_loc}/{self.start.constr_atoms[0]}{atom}/{calc}/control.in",
            "w",
        ) as control_file:
            control_file.writelines(control_content)

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

    def run_ground(self) -> None:
        """
        Run the ground state calculation.
        """

        ground_calc = du.GroundCalc(
            self.start.run_loc,
            self.start.atoms,
            self.start.basis_set,
            self.start.species,
            self.start.ase,
            self.start.hpc,
        )
        ground_calc.run(
            self.start.geom_inp,
            self.start.control_inp,
            self.start.constr_atom,
            self.start.calc,
            self.control_opts,
            self.l_vecs,
            self.start.print_output,
            self.start.nprocs,
            self.start.binary,
        )

    def setup_excited_calcs(self) -> Tuple[object, List[int]]:
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

        ground_geom = f"{self.start.run_loc}/ground/geometry.in"
        self.ground_control = f"{self.start.run_loc}/ground/control.in"

        # Ensure constrained atoms are defined
        if len(self.start.spec_at_constr) == 0 and self.start.constr_atom is None:
            raise click.MissingParameter(
                param_hint="-c/--constrained_atom or -s/--specific_atom_constraint",
                param_type="option",
            )

        # Convert constr_atoms to a list
        if isinstance(self.start.constr_atom, list) is False:
            constr_atoms = [self.start.constr_atom]
        else:
            constr_atoms = self.start.constr_atom

        # Create a list of element symbols to constrain
        if len(self.start.spec_at_constr) > 0:
            element_symbols = du.get_element_symbols(
                ground_geom, self.start.spec_at_constr
            )[0]
            constr_atoms = element_symbols
        else:
            element_symbols = constr_atoms

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

        self._check_params(self.start)

        # Check required arguments have been given
        du.check_args(("ks_range", self.ks_range))

        # TODO allow this for multiple constrained atoms using n_atoms
        for atom in element_symbols:
            fo.get_electronic_structure(atom)

        self._call_setups(fo)

        return fo, atom_specifier

    def pre_init_2(self, fo, atom_specifier) -> str:
        """
        Setup everything for the 2nd init calculation.

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

        self._check_params()

        for i in range(len(atom_specifier)):
            i += 1

            # Catch for if init_1 hasn't been run
            self._check_prev_runs("init_1", i)

            if len(self.control_opts) > 0 or self.start.control_inp:
                self._add_control_opts(fo, i, "init_2")

            # Copy the restart files to init_2 from init_1
            self._cp_restart_files(i, "init_1", "init_2")

        # Prevent SCF not converged errors from printing
        # It could be an issue to do this if any other errors occur
        spec_run_info = " 2>/dev/null"

        return spec_run_info

    def pre_hole(self, fo, atom_specifier) -> str:
        """
        Setup everything for the hole calculation.

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

        self._check_params(include_hpc=False)

        if self.start.hpc:
            du.check_args(("ks_range", self.ks_range))

            for atom in self.start.element_symbols:
                fo.get_electronic_structure(atom)

            # Setup files required for the initialisation and hole calculations
            proj = Projector(fo)
            self._call_setups(proj)

        # Add molecule identifier to hole geometry.in
        with open(
            f"{self.start.run_loc}/{self.start.constr_atoms[0]}{atom_specifier[0]}"
            "/hole/geometry.in",
            "r",
        ) as hole_geom:
            lines = hole_geom.readlines()

        lines.insert(4, f"# {self.start.spec_mol}\n")

        with open(
            f"{self.start.run_loc}/{self.start.constr_atoms[0]}{atom_specifier[0]}/hole"
            "/geometry.in",
            "w",
        ) as hole_geom:
            hole_geom.writelines(lines)

        if not self.start.hpc:
            for i in range(len(atom_specifier)):
                i += 1

                # Check for if init_2 hasn't been run
                self._check_prev_runs("init_2", i)

                if len(self.control_opts) > 0 or self.start.control_inp:
                    # Add any additional control options to the hole control file
                    self._add_control_opts(fo, i, "hole")

                # Copy the reself.start files to hole from init_2
                self._cp_restart_files(i, "init_2", "hole")

        # Don't redirect STDERR to /dev/null as not converged errors should not occur
        # here
        spec_run_info = ""

        return spec_run_info

    def run_excited(self, start, atom_specifier) -> None:
        """
        Run the projector calculations.

        Parameters
        ----------
            start : Start
                instance of the Start object
            atom_specifier : List[int]
                atom indices to constrain
        """

        # TODO: Move the below if statement to outside the function
        # if (
        #     self.run_type != "ground"
        #     and os.path.isfile(
        #         f"{self.run_loc}/{self.constr_atoms[0]}{atom_specifier[0]}"
        #         f"/{self.run_type}/aims.out"
        #     )
        #     is False
        #     and not self.hpc
        # ):
        # elif self.run_type != "ground" and not self.hpc:
        #     print(
        #         f"{self.run_type} calculations already completed, "
        #         "skipping calculation..."
        #     )

        du.set_env_vars()

        if start.print_output:  # Print live output of calculation
            for i in range(len(atom_specifier)):
                i += 1
                os.system(
                    f"cd {start.run_loc}/{start.constr_atoms[0]}{i}/{start.run_type} "
                    f"&& mpirun -n {start.nprocs} {start.binary} "
                    f"| tee aims.out {start.spec_run_info}"
                )
                if self.run_type != "init_1" and self.run_type != "init_2":
                    du.print_ks_states(start.run_loc)

        else:
            with click.progressbar(
                range(len(atom_specifier)),
                label=f"calculating {self.run_type}:",
            ) as prog_bar:
                for i in prog_bar:
                    i += 1
                    os.system(
                        f"cd {start.run_loc}/{start.constr_atoms[0]}{i}/"
                        f"{self.run_type} && mpirun -n {start.nprocs} {start.binary}"
                        " > aims.out {self.spec_run_info}"
                    )

            # TODO figure out how to parse STDOUT so a compeltely successful calculation
            # message can be given or not
            # print(f"{run_type} calculations completed successfully")


def basis_wrapper(
    ctx,
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
    """Force occupation of the basis states."""

    # It gets annoying to type the full context object out every time
    run_loc = ctx.obj["RUN_LOC"]
    geom = ctx.obj["GEOM_INP"]
    control = ctx.obj["CONTROL_INP"]
    spec_mol = ctx.obj["SPEC_MOL"]
    atoms = ctx.obj["ATOMS"]
    ase = ctx.obj["ASE"]
    species = ctx.obj["SPECIES"]
    basis_set = ctx.obj["BASIS_SET"]
    nprocs = ctx.obj["NPROCS"]
    binary = ctx.obj["BINARY"]
    hpc = ctx.obj["HPC"]
    constr_atoms = ctx.obj["CONSTR_ATOM"]
    spec_at_constr = ctx.obj["SPEC_AT_CONSTR"]
    occ = ctx.obj["OCC"]
    print_output = ctx.obj["PRINT"]

    if ase:
        calc = ctx.obj["CALC"]
    else:
        calc = None

    # Raise a warning if no additional control options have been specified
    if len(control_opts) < 1 and control is None:
        print(
            "\nWarning: no control options provided, using default options "
            "which can be found in the 'control.in' file"
        )

    # Convert control options to a dictionary
    control_opts = du.convert_opts_to_dict(control_opts, None)

    if run_type == "ground":
        du.ground_calc(
            run_loc,
            geom,
            control,
            atoms,
            None,
            basis_set,
            species,
            calc,
            ase,
            control_opts,
            constr_atoms,
            nprocs,
            binary,
            hpc,
            print_output,
        )

        # Ground must be run separately to hole calculations
        return

    else:  # run_type == 'hole'
        ground_geom = f"{run_loc}/ground/geometry.in"

    if len(spec_at_constr) == 0 and constr_atoms is None:
        raise click.MissingParameter(
            "No atoms have been specified to constrain, please provide either"
            " the -c/--constrained_atom or the -s/--specific_atom_constraint arguments",
            param_type="option",
        )

    if run_type == "hole":
        du.check_args(
            ("atom_index", atom_index),
            ("ks_max", ks_max),
            ("n_qn", n_qn),
            ("l_qn", l_qn),
            ("m_qn", m_qn),
        )

        if len(spec_at_constr) == 0 and len(constr_atoms) == 0:
            raise click.BadParameter(
                "no atoms have been specified to constrain, please use the "
                "-c/--constr_atoms or -s/--spec_at_constr options"
            )

        # Ensure that aims always runs with the following environment variables:
        os.system("export OMP_NUM_THREADS=1")
        os.system("export MKL_NUM_THREADS=1")
        os.system("export MKL_DYNAMIC=FALSE")
        os.system("ulimit -s unlimited")

        if os.path.isfile(f"{run_loc}/ground/aims.out") is False:
            raise FileNotFoundError(
                "\nERROR: ground aims.out not found, please ensure the ground calculation has been "
                "run"
            )

        # Convert constr_atoms to a list
        if isinstance(constr_atoms, list) is False:
            constr_atoms = [constr_atoms]

        # Create a list of element symbols to constrain
        if len(spec_at_constr) > 0:
            element_symbols = du.get_element_symbols(ground_geom, spec_at_constr)[0]
            constr_atoms = element_symbols
        else:
            element_symbols = constr_atoms

        # Create the directories required for the hole calculation
        fo = ForceOccupation(
            element_symbols,
            run_loc,
            ground_geom,
            control_opts,
            f"{species}/defaults_2020/{basis_set}",
        )

        # Get atom indices from the ground state geometry file
        atom_specifier = fo.read_ground_inp(constr_atoms, spec_at_constr, ground_geom)

        if (
            os.path.isfile(f"{run_loc}/{constr_atoms[0]}{atom_specifier[0]}/aims.out")
            is True
        ):
            print("hole calculations already completed, skipping calculation...")

        basis = Basis(fo)
        basis.setup_basis(
            multiplicity, n_qn, l_qn, m_qn, occ, ks_max, occ_type, basis_set, species
        )

        # Add molecule identifier to hole geometry.in
        with open(
            f"{run_loc}{constr_atoms[0]}{atom_specifier[0]}/geometry.in", "r"
        ) as hole_geom:
            lines = hole_geom.readlines()

        lines.insert(4, f"# {spec_mol}\n")

        with open(
            f"{run_loc}/{constr_atoms[0]}{atom_specifier[0]}/geometry.in", "w"
        ) as hole_geom:
            hole_geom.writelines(lines)

        # TODO allow multiple constraints using n_atoms

        for i in range(len(atom_specifier)):
            i += 1

            if len(control_opts) > 0 or control:
                # Add any additional control options to the hole control file
                parsed_control_opts = fo.get_control_keywords(
                    f"{run_loc}/{constr_atoms[0]}{i}/control.in"
                )
                mod_control_opts = fo.mod_keywords(control_opts, parsed_control_opts)
                control_content = fo.change_control_keywords(
                    f"{run_loc}/{constr_atoms[0]}{i}/control.in", mod_control_opts
                )

                with open(
                    f"{run_loc}/{constr_atoms[0]}{i}/control.in", "w"
                ) as control_file:
                    control_file.writelines(control_content)

        if not hpc:  # Run the hole calculation
            if print_output:  # Print live output of calculation
                for i in range(len(atom_specifier)):
                    i += 1
                    os.system(
                        f"cd {run_loc}/{constr_atoms[0]}{i} && mpirun -n "
                        f"{nprocs} {binary} | tee aims.out"
                    )
                    du.print_ks_states(run_loc)

            else:
                with click.progressbar(
                    range(len(atom_specifier)), label="calculating basis hole:"
                ) as prog_bar:
                    for i in prog_bar:
                        i += 1
                        os.system(
                            f"cd {run_loc}/{constr_atoms[0]}{i} && mpirun -n "
                            f"{nprocs} {binary} > aims.out"
                        )

        # These need to be passed to process()
        ctx.obj["RUN_TYPE"] = run_type
        ctx.obj["AT_SPEC"] = atom_specifier

    # Compute the dscf energies and plot if option provided
    if run_type == "hole":
        process(ctx)
