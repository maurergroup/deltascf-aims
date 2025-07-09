import glob
import os
import warnings

import yaml
from ase import Atoms

from deltascf_aims.utils.checks import check_species_in_control


def add_additional_basis(elements, content, target_atom) -> list[str]:
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

    current_path = os.path.dirname(os.path.realpath(__file__))

    # Get the additional basis set
    if "dscf_utils" in current_path.split("/"):
        with open(f"{current_path}/utils/add_basis_functions.yml") as f:
            ad_basis = yaml.safe_load(f)
    else:
        with open(f"{current_path}/utils/add_basis_functions.yml") as f:
            ad_basis = yaml.safe_load(f)

    if [*target_atom][-1][0] == "1":
        root_target_atom = "".join([*target_atom][:-1])
    else:
        root_target_atom = target_atom

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
        if "species" in line and target_atom in line:
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

    # Find the line which contains the appropriate row of '#'s after the species and element
    div_counter = 0
    insert_point = 0
    for i, line in enumerate(content[basis_def_start:]):
        if separator in line:
            if "species" in line and target_atom in line:
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

    warnings.warn("There was an error with adding the additional basis function")
    return content


def add_control_opts(
    start,
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

    parsed_control_opts = get_control_keywords(
        f"{start.run_loc}/{constr_atom}{i_atom}/{calc}/control.in"
    )
    mod_control_opts = mod_keywords(control_opts, parsed_control_opts)
    control_content = change_control_keywords(
        f"{start.run_loc}/{constr_atom}{i_atom}/{calc}/control.in",
        mod_control_opts,
    )

    with open(
        f"{start.run_loc}/{constr_atom}{i_atom}/{calc}/control.in",
        "w",
    ) as control_file:
        control_file.writelines(control_content)


def get_control_keywords(control) -> dict:
    """
    Get the keywords in a control.in file.

    Parameters
    ----------
    control : str
        path to the control file

    Returns
    -------
    opts : dict
        dictionary of keywords in the control file
    """
    # Find and replace keywords in control file
    with open(control) as read_control:
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


def mod_keywords(ad_cont_opts, opts) -> dict:
    """
    Update default or parsed keywords with user-specified keywords.

    Parameters
    ----------
    ad_cont_opts : dict
        User-specified keywords
    opts : dict
        Default keywords

    Returns
    -------
    opts : dict
        Keywords
    """
    for key in list(ad_cont_opts.keys()):
        opts.update({key: ad_cont_opts[key]})

    return opts


def change_control_keywords(control, opts) -> list[str]:
    """
    Modify the keywords in a control.in file from a dictionary of options.

    Parameters
    ----------
    control : str
        path to the control file
    opts : dict
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


def convert_opts_to_dict(opts: tuple[str], pbc: tuple[int] | None) -> dict:
    """
    Convert the control options from a tuple to a dictionary.

    Parameters
    ----------
    opts : tuple[str]
        tuple of control options
    pbc : tuple[int]
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


def convert_tuple_key_to_str(control_opts: dict) -> dict:
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

    return control_opts


def write_control(
    run_loc: str,
    control_opts: dict,
    atoms: Atoms,
    int_grid: str,
    add_extra_basis: bool,
    defaults: str,
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
    lines = change_control_keywords(f"{run_loc}/control.in", control_opts)

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
