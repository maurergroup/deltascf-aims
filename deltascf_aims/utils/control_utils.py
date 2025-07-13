from pathlib import Path
from typing import TYPE_CHECKING
from warnings import warn

import yaml
from ase import Atoms

from deltascf_aims.utils.checks_utils import check_species_in_control

if TYPE_CHECKING:
    from deltascf_aims.core import Start


def add_additional_basis(  # noqa: PLR0912
    elements: list[str], content: list[str], target_atom: str
) -> list[str]:
    """
    Insert an additional basis set to control.in.

    Parameters
    ----------
    elements : List[str]
        list of all supported elements
    content : List[str]
        list of lines in the control file
    target_atom : str
        element symbol of target atom

    Returns
    -------
    content : List[str]
        list of lines in the control file
    """
    # Check the additional functions haven't already been added to control
    for line in content:
        if "# Additional basis functions for atom with a core hole" in line:
            return content

    # Get the additional basis set
    with (Path(__file__).parent / "add_basis_functions.yml").open() as f:
        ad_basis = yaml.safe_load(f)

    root_target_atom = target_atom[:-1] if target_atom[-1] == "1" else target_atom

    try:
        el_ad_basis = ad_basis[root_target_atom]
    except KeyError:
        print(
            f"Warning: the additional basis set for {target_atom} is not yet supported."
            " The calculation will continue without the additional core-hole basis set."
        )
        el_ad_basis = ""

    # Add the additional basis set before the 5th line of '#'s after the species
    # and element line defining the start of the basis set. This is the only
    # consistent marker across all the basis sets after which the additional basis
    # set can be added.
    basis_def_start = 0
    for i, line in enumerate(content):
        if "species" in line and target_atom == line.split()[1]:
            basis_def_start = i
            break

    # Get the atomic number of the target atom
    atom_index = elements.index(str(root_target_atom)) + 1

    # prefix 0 if atom_index is less than 10
    if atom_index < 10:
        atom_index = f"0{atom_index}"

    # Append a separator to the end of the file
    # This helps with adding the additional basis set in the correct positions for
    # some basis sets.
    separator = 80 * "#"
    content.append(separator + "\n")

    # Find the line which contains the appropriate row of '#'s after the species and
    # element
    div_counter = 0
    insert_point = 0
    for i, line in enumerate(content[basis_def_start:]):
        if separator in line:
            if "species" in line and target_atom == line.split()[1]:
                break

            if div_counter == 3:
                insert_point = i + basis_def_start

            if div_counter < 4:
                div_counter += 1

            else:
                insert_point = i + basis_def_start
                break

    # Append the additional basis set to the end of the basis set in the control file
    if (
        el_ad_basis != ""
        and basis_def_start != 0
        and insert_point != 0
        and div_counter != 0
    ):
        content.insert(insert_point, f"{el_ad_basis}\n")
        return content

    warn("There was an error with adding the additional basis function", stacklevel=2)
    return content


def add_control_opts(
    start: "Start",
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

    control_in = start.run_loc.joinpath(f"{constr_atom}{i_atom}", calc, "control.in")

    parsed_control_opts = get_control_keywords(control_in)
    mod_control_opts = mod_keywords(control_opts, parsed_control_opts)
    control_content = change_control_keywords(control_in, mod_control_opts)

    with control_in.open("w") as f:
        f.writelines(control_content)


def get_control_keywords(control: Path) -> dict:
    """
    Get the keywords in a control.in file.

    Parameters
    ----------
    control : pathlib.Path
        path to the control file

    Returns
    -------
    opts : dict
        dictionary of keywords in the control file
    """
    # Find and replace keywords in control file
    with control.open() as read_control:
        content = read_control.readlines()

    # Get keywords
    opts = {}
    for line in content:
        spl = line.split()

        # Break when basis set definitions start
        if 80 * "#" in line:
            break

        # Add the dictionary value as a string
        if len(spl) > 1 and "#" not in spl[0]:
            if len(spl[1:]) > 1:
                opts[spl[0]] = " ".join(spl[1:])

            else:
                opts[spl[0]] = spl[1]

    return opts


def mod_keywords(ad_cont_opts: dict[str, str], opts: dict[str, str]) -> dict[str, str]:
    """
    Update default or parsed keywords with user-specified keywords.

    Parameters
    ----------
    ad_cont_opts : dict[str, str]
        User-specified keywords
    opts : dict[str, str]
        Default keywords

    Returns
    -------
    opts : dict[str, str]
        Keywords
    """
    for key in list(ad_cont_opts.keys()):
        opts.update({key: ad_cont_opts[key]})

    return opts


def change_control_keywords(control: Path, opts: dict[str, str]) -> list[str]:
    """
    Modify the keywords in a control.in file from a dictionary of options.

    Parameters
    ----------
    control : Path
        path to the control file
    opts : dict[str, str]
        dictionary of keywords to change

    Returns
    -------
    content : List[str]
        list of lines in the control file
    """
    # Find and replace keywords in control file
    with open(control) as read_control:
        content = read_control.readlines()

    divider_1 = "#" + 79 * "="
    divider_2 = 80 * "#"

    short_circuit = False

    # Change keyword lines
    for i, opt in enumerate(opts):
        ident = 0

        for j, line in enumerate(content):
            if short_circuit:
                break

            spl = line.split()

            if opt in spl:
                content[j] = f"{opt:<34} {opts[opt]}\n"
                break

            # Marker if ASE was used so keywords will be added in the same
            # place as the others
            if divider_1 in line:
                ident += 1
                if ident == 3:
                    content.insert(j, f"{opt:<34} {opts[opt]}\n")
                    break

            # Marker for non-ASE input files so keywords will be added
            # before the basis set definitions
            elif divider_2 in line:
                ident += 1
                content.insert(j - 1, f"{opt:<34} {opts[opt]}\n")
                break

        # Ensure keywords are added in all other instances
        if ident == 0:
            short_circuit = True

            if i == 0:
                content.append(divider_1 + "\n")
                content.append(f"{opt:<34} {opts[opt]}\n")

            else:
                content.append(f"{opt:<34} {opts[opt]}\n")

            if i == len(opts) - 1:
                content.append(divider_1 + "\n")

    return content


def convert_opts_to_dict(
    opts: tuple[str, ...], pbc: tuple[int, int, int] | None
) -> dict:
    """
    Convert the control options from a tuple to a dictionary.

    Parameters
    ----------
    opts : tuple[str, ...]
        tuple of control options
    pbc : tuple[int, int, int]
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


def convert_tuple_key_to_str(control_opts: dict[str, str | tuple]) -> dict[str, str]:
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

    return control_opts  # pyright: ignore[reportReturnType]


def write_control(
    run_loc: Path,
    control_opts: dict,
    atoms: Atoms,
    int_grid: str,
    defaults: Path,
    add_extra_basis: bool,
    constr_atom: str | None = None,
) -> None:
    """
    Write a control.in file.

    Parameters
    ----------
    run_loc : pathlib.Path
        Path to the calculation directory
    control_opts : dict[str, str]
        Dictionary of control options
    atoms : Atoms
        ASE atoms object
    int_grid : str
        Basis set density
    defaults : Path
        Path to the species_defaults directory
    add_extra_basis : bool
        Whether to add extra core-hole augmented basis functions basis set
    constr_atom : str | None, default = None
        Constrained atom symbol - only needed if add_extra_basis is True
    """
    # Firstly create the control file if it doesn't exist
    control_in = run_loc / "control.in"

    if not control_in.is_file():
        control_in.touch()

    control_opts = convert_tuple_key_to_str(control_opts)
    lines = change_control_keywords(control_in, control_opts)

    with control_in.open("w") as f:
        f.writelines(lines)

    # Then add the basis set
    elements = list(set(atoms.get_chemical_symbols()))

    for el in elements:
        # Add extra basis functions
        if add_extra_basis and constr_atom is not None:
            with control_in.open() as c:
                control_content = c.readlines()

            current_path = Path(__file__).resolve().parent

            with (current_path / "elements.yml").open() as yf:
                elements = yaml.load(yf, Loader=yaml.SafeLoader)

            new_content = add_additional_basis(elements, control_content, constr_atom)

            with control_in.open("w") as c:
                # Write the new content to the control.in file
                c.writelines(new_content)

        if not check_species_in_control(lines, el):
            basis_sets = list(
                (defaults / "defaults_2020" / int_grid).glob(f"*{el}_default")
            )

            if len(basis_sets) != 0:
                basis_set = basis_sets[0]
                with basis_set.open() as src, (run_loc / "control.in").open("a") as dst:
                    dst.write(src.read())

    # Copy it to the ground directory
    ground_dir = run_loc / "ground"
    ground_dir.mkdir(exist_ok=True)
    (run_loc / "control.in").replace(ground_dir / "control.in")
