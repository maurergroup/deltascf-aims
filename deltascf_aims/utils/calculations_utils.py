import glob
import os
from pathlib import Path
from typing import Literal
from warnings import warn

import numpy as np
import yaml
from ase.calculators.aims import Aims
from ase.io import write
from click import progressbar

from deltascf_aims.utils.checks_utils import check_spin_polarised
from deltascf_aims.utils.control_utils import (
    add_additional_basis,
    add_control_opts,
    write_control,
)


def create_calc(
    procs: int, binary: Path, aims_cmd: str, species: Path, int_grid: str
) -> Aims:
    """
    Create an ASE calculator object.

    Parameters
    ----------
    procs : int
        number of processors to use
    binary : Path
        path to aims binary
    aims_cmd : str
        command to run aims
    species : Path
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
    spin_polarised = check_spin_polarised(lines)

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

    if platform in {"linux", "linux2"}:
        os.system("ulimit -s unlimited")
    elif platform == "darwin":
        os.system("ulimit -s hard")
    else:
        warn("OS not supported, please ensure ulimit is set to unlimited")


def warn_no_extra_control_opts(opts: dict, inp: Path | None) -> None:
    """
    Raise a warning if no additional control options have been specified.

    Parameters
    ----------
    opts : dict
        additional control options to be added to the control.in file
    inp : pathlib.Path | None
        path to custom control.in file

    """
    if len(opts) < 1 and inp is None:
        warn(
            "No extra control options provided, using default options which can be "
            "found in the 'control.in' file",
            stacklevel=2,
        )


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

        new_content = add_additional_basis(elements, control_content, constr_atom)

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
                if run_type not in {"init_1", "init_2"}:
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
