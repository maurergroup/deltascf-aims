import os
import resource
import shutil
import subprocess
from pathlib import Path
from sys import platform
from typing import TYPE_CHECKING, Literal
from warnings import warn

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

if TYPE_CHECKING:
    from ase.atoms import Atoms

    from deltascf_aims.core import Start


def create_calc(
    procs: int, binary: Path, aims_cmd: str, species: Path, int_grid: str
) -> Aims:
    """
    Create an ASE calculator object.

    Parameters
    ----------
    procs : int
        number of processors to use
    binary : pathlib.Path
        path to aims binary
    aims_cmd : str
        command to run aims
    species : pathlib.Path
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


def print_ks_states(run_loc: Path) -> None:  # noqa: PLR0912
    """
    Print the Kohn-Sham eigenvalues from a calculation.

    Parameters
    ----------
    run_loc : pathlib.Path
        Path to the calculation directory

    Raises
    ------
    ValueError
        Could not find the KS states
    """
    with (run_loc / "aims.out").open() as aims:
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
    """Set environment variables and ulimit for running FHI-aims."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["MKL_DYNAMIC"] = "FALSE"

    if platform in {"linux", "linux2"}:
        # Set stack size to unlimited
        resource.setrlimit(
            resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
        )
    elif platform == "darwin":
        # Set stack size to hard limit
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))
    else:
        warn(
            f"OS '{platform}' not supported, please ensure ulimit is set to unlimited",
            stacklevel=2,
        )


def warn_no_extra_control_opts(opts: dict, inp: Path | None) -> None:
    """
    Raise a warning if no additional control options have been specified.

    Parameters
    ----------
    opts : dict
        Additional control options to be added to the control.in file
    inp : pathlib.Path | None
        Path to custom control.in file

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
    run_loc : pathlib.Path
        Path to the calculation directory
    atoms : Atoms
        ASE atoms object
    basis_set : str
        Basis set density
    species : pathlib.Path
        Path to species directory
    ase : bool
        Whether to use ASE
    hpc : bool
        Whether to only write input files and not run the calculation

    Methods
    -------
    setup_ground(geometry_in, control_in, control_opts, start)
        Setup the ground calculation files and directories
    add_extra_basis_fns(constr_atom)
        Add additional basis functions to the basis set
    run_ground(control_opts, add_extra_basis, l_vecs, print_output, nprocs, binary, calc)
        Run the ground state calculation
    """

    def __init__(
        self,
        run_loc: Path,
        atoms: "Atoms",
        basis_set: str,
        species: Path,
        ase: bool,
        hpc: bool,
    ):
        self.run_loc: Path = run_loc
        self.atoms = atoms
        self.basis_set = basis_set
        self.species = species
        self.ase = ase
        self.hpc = hpc

    def setup_ground(
        self,
        geometry_in: Path | None,
        control_in: Path,
        control_opts: dict[str, str],
        start: "Start",
    ) -> None:
        """
        Set up the ground calculation files and directories.

        Parameters
        ----------
        geometry_in : Path |None
            path to the geometry.in file
        control_in : str
            path to the control.in file
        control_opts : dict[str, str]
            additional options to be added to the control.in file
        start : Start
            instance of Start class
        """
        # Create the ground directory if it doesn't already exist
        (self.run_loc / "ground").mkdir(exist_ok=True)

        # Write the geometry file if the system is specified through CLI
        if geometry_in is None:
            write(self.run_loc / "geometry.in", self.atoms, format="aims")

        # Copy the geometry.in and control.in files to the ground directory
        if control_in is not None:
            shutil.copy2(control_in.name, self.run_loc / "ground")

            # Add any additional options to the control file
            if len(control_opts) > 0:
                add_control_opts(start, "", "", "ground", control_opts)

        if geometry_in is not None:
            shutil.copy2(geometry_in.name, self.run_loc / "ground")

    @staticmethod
    def add_extra_basis_fns(constr_atom: str, control_in: Path) -> None:
        """
        Add additional basis functions to the basis set.

        Parameters
        ----------
        constr_atom : str
            element symbol of the constrained atom
        control_in : pathlib.Path
            Path to the control.in file
        """
        with control_in.open() as c:
            control_content = c.readlines()

        current_path = Path(__file__).resolve().parent

        with (current_path / "elements.yml").open() as yf:
            elements = yaml.load(yf, Loader=yaml.SafeLoader)

        new_content = add_additional_basis(elements, control_content, constr_atom)

        with control_in.open("w") as control:
            # Write the new content to the control.in file
            control.writelines(new_content)

    def _with_ase(
        self,
        calc: Aims,
        control_opts: dict[str, str],
        add_extra_basis: bool,
        constr_atom: str,
    ) -> None:
        """
        Run the ground state calculation using ASE.

        Parameters
        ----------
        calc : Aims
            FHI-aims calculator instance
        control_opts : dict[str, str]
            Control options
        add_extra_basis : bool
            Whether to add extra basis sets for a core hole
        constr_atom : str
            Constrained atom symbol
        """
        # Change the defaults if any are specified by the user
        # Update with all control options from the calculator
        calc.set(**control_opts)
        control_opts = calc.parameters

        if self.hpc:
            # Prevent species dir from being written
            control_opts.pop("species_dir")

            print("writing geometry.in file...")
            write(self.run_loc / "ground/geometry.in", images=self.atoms, format="aims")

            print("writing control.in file...")
            write_control(
                self.run_loc,
                control_opts,
                self.atoms,
                self.basis_set,
                self.species,
                add_extra_basis,
                constr_atom,
            )

        else:
            print("running calculation...")
            self.atoms.get_potential_energy()

            # Move files to ground directory
            shutil.copy2(self.run_loc / "geometry.in", self.run_loc / "ground")
            shutil.copy2(self.run_loc / "control.in", self.run_loc / "ground")
            shutil.move(self.run_loc / "aims.out", self.run_loc / "ground")
            shutil.move(self.run_loc / "parameters.ase", self.run_loc / "ground")

    def _without_ase(
        self, print_output: bool, aims_cmd: str, nprocs: int, binary: Path
    ) -> None:
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
        binary : Path
            Path to the FHI-aims binary
        """
        print("running calculation...")

        ground_dir = self.run_loc / "ground"
        cmd = [str(aims_cmd), str(nprocs), str(binary)]

        if print_output:  # Show live output of calculation
            with (ground_dir / "aims.out").open("w") as outfile:
                process = subprocess.Popen(
                    cmd,
                    cwd=ground_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                for line in process.stdout:
                    print(line, end="")
                    outfile.write(line)

                process.wait()
        else:
            with (ground_dir / "aims.out").open("w") as outfile:
                subprocess.run(
                    cmd,
                    cwd=ground_dir,
                    stdout=outfile,
                    stderr=subprocess.STDOUT,
                    check=True,
                    text=True,
                )

    def run_ground(
        self,
        control_opts: dict[str, str],
        add_extra_basis: bool,
        constr_atom: str,
        print_output: bool,
        aims_cmd: str,
        nprocs: int,
        binary: Path,
        calc: Aims,
    ) -> None:
        """
        Run the ground state calculation.

        Parameters
        ----------
        control_opts : dict[str, str]
            Control options
        add_extra_basis : bool
            Whether to add additional basis function to the core hole
        constr_atom : str
            Constrained atom symbol
        print_output : bool
            Whether to print the output of the calculation
        aims_cmd : str
            Command to run FHI-aims
        nprocs : int
            Number of processors to use with the parallel command
        binary : Path
            Path to the FHI-aims binary
        calc : Aims
            Instance of an ASE calculator object
        """
        if not self.hpc:  # Run the ground state calculation
            set_env_vars()

        if self.ase:  # Use ASE
            self._with_ase(calc, control_opts, add_extra_basis, constr_atom)

        elif not self.hpc:  # Don't use ASE
            self._without_ase(print_output, aims_cmd, nprocs, binary)

        # Print the KS states from aims.out so it is easier to specify the
        # KS states for the hole calculation
        if not self.hpc:
            print_ks_states(self.run_loc / "ground")


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

    def __init__(self, start: "Start"):
        self.start = start

    def check_restart_files(self, constr_atom: str, prev_calc: str, atom: int) -> None:
        """
        Check if the restart files from the previous calculation exist.

        Parameters
        ----------
        constr_atom : str
            Constrained atom
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
        restart_files = list(
            (self.start.run_loc / f"{constr_atom}{atom}" / prev_calc).glob("*restart*")
        )
        if len(restart_files) < 1:
            print(
                f'{prev_calc} restart files not found, please ensure "{prev_calc}"'
                " has been run"
            )
            raise FileNotFoundError

    def check_prereq_calc(
        self,
        current_calc: Literal["init_1", "init_2", "hole"],
        constr_atom: str,
        constr_method: Literal["projector", "basis"],
    ) -> Literal["ground", "init_1", "init_2"]:
        """
        Check if the prerequisite calculation has been run.

        Parameters
        ----------
        current_calc : Literal["init_1", "init_2", "hole"]
            Type of excited calculation to check for
        constr_atom : str
            Constrained atom
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
        search_path = None

        match current_calc:
            case "init_1":
                prev_calc = "ground"
                search_path = self.start.run_loc / "ground" / "aims.out"

            case "init_2":
                prev_calc = "init_1"
                try:
                    search_path = next(
                        self.start.run_loc.glob(f"{constr_atom}*/init_1/aims.out")
                    )
                except StopIteration as err:
                    msg = (
                        f"aims.out for {prev_calc} not found, please ensure the "
                        f"{prev_calc} calculation has been run"
                    )
                    raise FileNotFoundError(msg) from err

            case "hole":
                if constr_method == "projector" and not self.start.hpc:
                    prev_calc = "init_2"
                    try:
                        search_path = next(
                            self.start.run_loc.glob(f"{constr_atom}*/init_2/aims.out")
                        )
                    except StopIteration as err:
                        msg = (
                            f"aims.out for {prev_calc} not found, please ensure the "
                            f"{prev_calc} calculation has been run"
                        )
                        raise FileNotFoundError(msg) from err

                if constr_method == "projector" and self.start.hpc:
                    prev_calc = "ground"
                    search_path = self.start.run_loc / "ground" / "aims.out"

                if constr_method == "basis":
                    prev_calc = "ground"
                    search_path = self.start.run_loc / "ground" / "aims.out"

        if prev_calc is None:
            raise TypeError("Unable to determine the previous calculation run")

        if search_path is None or not search_path.is_file():
            msg = (
                f"aims.out for {prev_calc} not found, please ensure the "
                f"{prev_calc} calculation has been run"
            )
            raise FileNotFoundError(msg)

        return prev_calc

    def run_excited(
        self,
        atom_specifier: list[int],
        constr_atom: str,
        run_type: Literal["init_1", "init_2", "hole"],
        basis_constr: bool = False,
    ) -> None:
        """
        Run an excited state calculation.

        Parameters
        ----------
        atom_specifier : list[int]
            list of atom indices to constrain
        constr_atom : str
            Constrained atom
        run_type : Literal["init_1", "init_2", "hole"]
            Type of excited calculation to run
        basis_constr : bool, default=False
            Whether the calculation uses the basis occupation constraint method
        """
        # Don't cd into hole for basis calculation
        run_type_dir = Path() if basis_constr else Path(run_type)

        set_env_vars()

        if self.start.print_output:  # Print live output of calculation
            for i in range(len(atom_specifier)):
                work_dir = (
                    self.start.run_loc
                    / f"{constr_atom}{atom_specifier[i]}"
                    / run_type_dir
                )
                cmd = [
                    str(self.start.run_cmd),
                    str(self.start.nprocs),
                    str(self.start.binary),
                ]
                # Open aims.out for writing and tee output to both stdout and file
                with (work_dir / "aims.out").open("w") as outfile:
                    process = subprocess.Popen(
                        cmd,
                        cwd=work_dir,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                    for line in process.stdout:
                        print(line, end="")
                        outfile.write(line)
                    process.wait()

                if run_type not in {"init_1", "init_2"}:
                    print_ks_states(work_dir)

        else:
            with progressbar(
                range(len(atom_specifier)),
                label=f"calculating {run_type}:",
            ) as prog_bar:
                for i in prog_bar:
                    work_dir = (
                        self.start.run_loc
                        / f"{constr_atom}{atom_specifier[i]}"
                        / run_type_dir
                    )
                    cmd = [
                        str(self.start.run_cmd),
                        str(self.start.nprocs),
                        str(self.start.binary),
                    ]
                    with (work_dir / "aims.out").open("w") as outfile:
                        subprocess.run(
                            cmd,
                            cwd=work_dir,
                            stdout=outfile,
                            stderr=subprocess.STDOUT,
                            check=True,
                            text=True,
                        )

            # TODO figure out how to parse STDOUT so a completed successfully calculation
            # message can be given or not
            # print(f"{run_type} calculations completed successfully")
