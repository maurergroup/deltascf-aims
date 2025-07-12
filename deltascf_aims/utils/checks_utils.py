from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from warnings import warn

from click import BadParameter, MissingParameter

if TYPE_CHECKING:
    from deltascf_aims.core import Start


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
                arg = list(arg)  # noqa: PLW2901
                arg[0] = "molecule"
                arg = tuple(arg)  # noqa: PLW2901

            raise MissingParameter(param_hint=f"'--{arg[0]}'", param_type="option")


def check_constrained_geom(geom_file: Path) -> None:
    """
    Check for `constrain_relaxation` keywords in the geometry.in.

    Parameters
    ----------
    geom_file : pathlib.Path
        Path to geometry.in file

    Raises
    ------
    SystemExit
        Exit the program if the `constrain_relaxation` keyword is found
    """
    with geom_file.open() as f:
        for line in f:
            if "constrain_relaxation" in line:
                print("`constrain_relaxation` keyword found in geometry.in")
                print("ensure that no atoms are fixed in the geometry.in file")
                print(
                    "the geometry of the structure should have already been relaxed "
                    "before running the single-point calculations"
                )
                print("aborting...")
                raise SystemExit(1)


def check_curr_prev_run(
    run_type: Literal["ground", "init_1", "init_2", "hole"],
    run_loc: Path,
    constr_atom: str,
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
    run_loc : Path
        Path to the calculation directory
    constr_atoms : str
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
        search_path = run_loc / run_type / "aims.out"
    elif constr_method == "projector":
        search_path = (
            run_loc / f"{constr_atom}{atom_specifier[0]}" / run_type / "aims.out"
        )
    elif constr_method == "basis":
        search_path = run_loc / f"{constr_atom}{atom_specifier[0]}" / "aims.out"

    if search_path.is_file() and not hpc and not force:
        warn("Calculation has already been completed", stacklevel=2)
        cont = None
        while cont != "y":
            cont = str(input("Do you want to continue? (y/n) ")).lower()

            if cont == "n":
                print("aborting...")
                raise SystemExit(1)

            if cont == "y":
                break


def check_k_grid(control_file: Path) -> bool:
    """
    Check if there is a k_grid in the control.in.

    Parameters
    ----------
    control_file : pathlib.Path
        Path to control.in file

    Returns
    -------
    k_grid : bool
        Whether the k_grid input parameter is found
    """
    k_grid = False

    with control_file.open() as f:
        for line in f:
            spl = line.split()
            if len(spl) > 0 and spl[0] == "k_grid":
                k_grid = True

    return k_grid


def check_lattice_vecs(geom_file: Path) -> bool:
    """
    Check if lattice vectors are given in the geometry.in file.

    Parameters
    ----------
    geom_file : pathlib.Path
        Path to geometry.in file

    Returns
    -------
    l_vecs : bool
        True if lattice vectors are found, False otherwise
    """
    l_vecs = False
    for line in geom_file.open():
        if "lattice_vector" in line:
            l_vecs = True

    return l_vecs


def check_params(start: "Start", include_hpc: bool = False) -> None:
    """
    Check that the parameters given in Start are valid.

    Parameters
    ----------
    start
        Instance of the Start class
    include_hpc : bool, default=False
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

    if start.constr_atom in [None, ""]:
        raise MissingParameter(
            param_hint="-c/--constrained_atom",
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


def check_spin_polarised(lines: list[str]) -> bool:
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
