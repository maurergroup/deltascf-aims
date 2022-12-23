"""Automate creation of files for FOP calculations in FHI-aims."""

import glob
import os
import shutil
import subprocess

import numpy as np
import yaml


def read_ground_inp(target_atom, geometry_path, **kwargs):
    """Find number of atoms in geometry."""

    # All supported elements
    with open("elements.yml", "r") as elements:
        element_symbols = yaml.load(elements, Loader=yaml.SafeLoader)

    # Check arguments and add atoms to list
    if target_atom not in element_symbols:
        raise ValueError("invalid element symbol")

    atoms = kwargs.get("atoms", None)

    atom_specifier = []

    if atoms is not None:
        if type(atoms) == list:
            for atom in atoms:
                if type(atom) is int:
                    atom_specifier.append(atom)
                else:
                    raise ValueError("atoms list should only contain integer values")

        elif type(atoms) == int:
            atom_specifier.append(atoms)

        else:
            raise ValueError("atoms list must be a list of ints or an int")

        print("specified atoms:", atom_specifier)
        return element_symbols, atom_specifier

    # Default to all atoms if specific atoms aren't specified
    elif len(atom_specifier) == 0:
        print(
            "atoms argument not specified, defaulting to all target atoms in geometry.in"
        )

        with open(geometry_path, "r") as geom_in:
            atom_counter = 0

            for line in geom_in:
                spl = line.split()

                if len(spl) > 0 and "atom" == spl[0] and target_atom in line:
                    atom_counter += 1
                    element = spl[-1]  # Identify atom
                    identifier = spl[0]  # Extra check that line is an atom

                    if identifier == "atom" and element == target_atom:
                        atom_specifier.append(atom_counter)

        print("specified atoms:", atom_specifier)

        return element_symbols, atom_specifier

    else:
        raise ValueError("invalid atoms argument")


def get_electronic_structure(element_symbols, target_atom):
    """Get valence electronic structure of target atom."""
    # Adapted from scipython.com question P2.5.12

    atom_index = element_symbols.index(str(target_atom)) + 1

    # Letters identifying subshells
    l_letter = ["s", "p", "d", "f", "g"]

    def get_config_str(config):
        """Turn a list of orbital, nelec pairs into a configuration string."""
        return ".".join(["{:2s}{:d}".format(*e) for e in config])

    # Create and order a list of tuples, (n+l, n, l), corresponding to the order
    # in which the corresponding orbitals are filled using the Madelung rule.
    nl_pairs = []
    for n in range(1, 8):
        for l in range(n):
            nl_pairs.append((n + l, n, l))
    nl_pairs.sort()

    # inl indexes the subshell in the nl_pairs list; nelec is the number of
    # electrons currently inhabiting this subshell
    inl, nelec = 0, 0
    # start with the 1s orbital
    n, l = 1, 0
    # a list of orbitals and the electrons they contain for a configuration
    config = [["1s", 0]]
    # Store the most recent Noble gas configuration encountered in a tuple with the
    # corresponding element symbol
    noble_gas_config = ("", "")

    s_config = "please.work"

    for i, _ in enumerate(element_symbols[:atom_index]):
        nelec += 1

        if nelec > 2 * (2 * l + 1):
            # this subshell is now full
            if l == 1:
                # The most recent configuration was for a Noble gas: store it
                noble_gas_config = (
                    get_config_str(config),
                    "[{}]".format(element_symbols[i - 1]),
                )
            # Start a new subshell
            inl += 1
            _, n, l = nl_pairs[inl]
            config.append(["{}{}".format(n, l_letter[l]), 1])
            nelec = 1
        else:
            # add an electron to the current subshell
            config[-1][1] += 1

        # Turn config into a string
        s_config = get_config_str(config)
        # Replace the heaviest Noble gas configuration with its symbol
        s_config = s_config.replace(*noble_gas_config)
        # print('{:2s}: {}'.format(element, s_config))

    output = list(s_config.split(".").pop(-1))
    valence = f"    valence      {output[0]}  {output[1]}   {output[2]}.1\n"

    return atom_index, valence


def setup_init_1(
    basis_set,
    defaults,
    target_atom,
    num_atom,
    calc_path,
    at_num,
    atom_valence,
):
    """Write new init directories and control files to calculate FOP."""

    iter_limit = "# sc_iter_limit           1\n"
    init_iter = "# sc_init_iter          75\n"
    ks_method = "KS_method                 serial\n"
    restart_file = "restart_write_only        restart_file\n"
    restart_save = "restart_save_iterations   20\n"
    restart_force = "# force_single_restartfile  .true.\n"
    charge = "charge                    0.1\n"
    output_cube = "#output                  cube spin_density\n"
    output_mull = "# output                  mulliken\n"
    output_hirsh = "# output                  hirshfeld\n"

    # Ensure returned variables are bound
    nucleus = "0"
    n_index = 0
    valence_index = 0

    # Check that specified basis_set is valid
    basis_set_opts = ["light", "intermediate", "tight", "really_tight"]

    if basis_set not in basis_set_opts:
        raise ValueError(
            f"defined basis set is not a valid option, available options are: \n{basis_set_opts}"
        )

    shutil.copyfile(
        f"{calc_path}ground/control.in", f"{calc_path}ground/control.in.new"
    )

    basis_set = glob.glob(
        f"{defaults}/defaults_2020/{basis_set}/*{target_atom}_default"
    )
    bash_add_basis = f"cat {basis_set[0]}"
    new_control = open(f"{calc_path}ground/control.in.new", "a")
    subprocess.run(bash_add_basis.split(), check=True, stdout=new_control)

    if type(num_atom) == list:
        for atom in num_atom:
            if type(atom) is not int:
                raise ValueError("num_atom must be an integer or list of integers")

        loop_iterator = num_atom

    elif type(num_atom) == int:
        loop_iterator = range(num_atom)
    else:
        raise ValueError("num_atom must be an integer or list of integers")

    for i in loop_iterator:
        if type(num_atom) != list:
            i += 1

        os.makedirs(f"{calc_path}{target_atom}{i}/init_1", exist_ok=True)
        shutil.copyfile(
            f"{calc_path}ground/control.in.new",
            f"{calc_path}{target_atom}{i}/init_1/control.in",
        )
        shutil.copyfile(
            f"{calc_path}ground/geometry.in",
            f"{calc_path}{target_atom}{i}/init_1/geometry.in",
        )

        found_target_atom = False
        control = f"{calc_path}{target_atom}{i}/init_1/control.in"
        geometry = f"{calc_path}{target_atom}{i}/init_1/geometry.in"

        # Change geometry file
        with open(geometry, "r") as read_geom:
            geom_content = read_geom.readlines()

        # Change atom to {atom}{num}
        atom_counter = 0
        for j, line in enumerate(geom_content):
            spl = line.split()

            if "atom" in line and target_atom in line:
                if atom_counter + 1 == i:
                    partial_hole_atom = f" {target_atom}1\n"
                    geom_content[j] = " ".join(spl[0:-1]) + partial_hole_atom

                atom_counter += 1

        with open(geometry, "w+") as write_geom:
            write_geom.writelines(geom_content)

        # Change control file
        with open(control, "r") as read_control:
            control_content = read_control.readlines()

        # Replace specific lines
        for j, line in enumerate(control_content):
            spl = line.split()

            if len(spl) > 1:
                # Fix basis sets
                if "species" == spl[0] and target_atom == spl[1]:
                    if found_target_atom is False:
                        control_content[j] = f"  species        {target_atom}1\n"
                        found_target_atom = True

                # Change keyword lines
                if "sc_iter_limit" in spl:
                    control_content[j] = iter_limit
                if "sc_init_iter" in spl:
                    control_content[j] = init_iter
                if "KS_method" in spl:
                    control_content[j] = ks_method
                if "restart_write_only" in spl:
                    control_content[j] = restart_file
                if "restart_save_iterations" in spl:
                    control_content[j] = restart_save
                if "force_single_restartfile" in spl:
                    control_content[j] = restart_force
                if "charge" in spl:
                    control_content[j] = charge
                if "#" == spl[0] and "charge" == spl[1]:
                    control_content[j] = charge
                if "cube spin_density" in spl:
                    control_content[j] = output_cube
                if "output" == spl[0] and "mulliken" == spl[1]:
                    control_content[j] = output_mull
                if "output" == spl[0] and "hirshfeld" == spl[1]:
                    control_content[j] = output_hirsh

        # Check if parameters not found
        no_iter_limit = False
        no_ks = False
        no_restart = False
        no_charge = False
        no_cube = False

        if iter_limit not in control_content:
            no_iter_limit = True
        if ks_method not in control_content:
            no_ks = True
        if restart_file not in control_content:
            no_restart = True
        if charge not in control_content:
            no_charge = True
        if output_cube not in control_content:
            no_cube = True

        # Write the data to the file
        with open(control, "w+") as write_control:
            write_control.writelines(control_content)

            # Append parameters to end of file if not found
            if no_iter_limit is True:
                write_control.write(iter_limit)
            if no_ks is True:
                write_control.write(ks_method)
            if no_restart is True:
                write_control.write(restart_file)
            if no_charge is True:
                write_control.write(charge)
            if no_cube is True:
                write_control.write(output_cube)

        # Add 0.1 charge
        with open(control, "r") as read_control:
            control_content = read_control.readlines()

        # Replace specific lines
        for j, line in enumerate(control_content):
            spl = line.split()

            if target_atom + "1" in spl:
                # Add to nucleus
                if f"    nucleus             {at_num}\n" in control_content[j:]:
                    n_index = (
                        control_content[j:].index(f"    nucleus             {at_num}\n")
                        + j
                    )
                    nucleus = control_content[n_index]  # save for hole
                    control_content[n_index] = f"    nucleus             {at_num}.1\n"
                elif f"    nucleus      {at_num}\n" in control_content[j:]:
                    n_index = (
                        control_content[j:].index(f"    nucleus      {at_num}\n") + j
                    )
                    nucleus = control_content[n_index]  # save for hole
                    control_content[n_index] = f"    nucleus      {at_num}.1\n"

                # Add to valence orbital
                if "#     ion occupancy\n" in control_content[j:]:
                    vbs_index = (
                        control_content[j:].index("#     valence basis states\n") + j
                    )
                    io_index = control_content[j:].index("#     ion occupancy\n") + j

                    # Check which orbital to add 0.1 to
                    principle_qns = np.array([])
                    azimuthal_orbs = np.array([])
                    azimuthal_qns = np.zeros(io_index - vbs_index - 1)
                    azimuthal_refs = {"s": 1, "p": 2, "d": 3, "f": 4}

                    # Get azimuthal and principle quantum numbers
                    for count, valence_orbital in enumerate(
                        control_content[vbs_index + 1 : io_index]
                    ):
                        principle_qns = np.append(
                            principle_qns,
                            np.array(valence_orbital.split()[1]),
                        ).astype(int)
                        azimuthal_orbs = np.append(
                            azimuthal_orbs,
                            np.array(valence_orbital.split()[2]),
                        )
                        azimuthal_qns[count] = azimuthal_refs[azimuthal_orbs[count]]
                        azimuthal_qns = azimuthal_qns.astype(int)

                    # Find the orbital with highest principle and azimuthal qn
                    highest_n = np.amax(principle_qns)
                    highest_n_index = np.where(principle_qns == highest_n)

                    # Check for highest l if 2 orbitals have the same n
                    if len(highest_n_index[0]) > 1:
                        highest_l = np.amax(azimuthal_qns)
                        highest_l_index = np.where(azimuthal_qns == highest_l)
                        addition_state = np.intersect1d(
                            highest_n_index, highest_l_index
                        )[0]
                    else:
                        addition_state = highest_n_index[0][0]

                    # Add the 0.1 electron
                    # valence = control_content[
                    #     vbs_index + addition_state + 1
                    # ]  # save for write hole file
                    valence_index = vbs_index + addition_state + 1
                    control_content[valence_index] = atom_valence
                    break

        with open(control, "w+") as write_control:
            write_control.writelines(control_content)

    print("init_1 files written successfully")

    return nucleus, n_index, valence_index


def setup_init_2(
    ks_states,
    calc_path,
    target_atom,
    num_atom,
    at_num,
    atom_valence,
    n_index,
    valence_index,
    occ_type,
):
    """Write new init directories and control files to calculate FOP."""

    if (
        type(ks_states) is not list
        or len(ks_states) != 2
        or not all(isinstance(i, int) for i in ks_states)
    ):
        raise ValueError("ks_states must be an integer list of length 2")

    iter_limit = "sc_iter_limit             1\n"
    restart_file = "restart             restart_file\n"
    restart_force = "# force_single_restartfile .true.\n"
    charge = "charge                    1.1\n"

    fop = None
    if occ_type == "old_projector":
        fop = f"force_occupation_projector {ks_states[0]} 1 0.0 {ks_states[0]} {ks_states[1]}\n"
    # elif p_type == 'new_fop':
    #     fop =

    if type(num_atom) == list:
        loop_iterator = num_atom
    else:
        loop_iterator = range(num_atom)

    for i in loop_iterator:
        if type(num_atom) != list:
            i += 1

        os.makedirs(f"{calc_path}{target_atom}{i}/init_2", exist_ok=True)
        shutil.copyfile(
            f"{calc_path}ground/control.in.new",
            f"{calc_path}{target_atom}{i}/init_2/control.in",
        )
        shutil.copyfile(
            f"{calc_path}{target_atom}{i}/init_1/geometry.in",
            f"{calc_path}{target_atom}{i}/init_2/geometry.in",
        )

        found_target_atom = False
        control = f"{calc_path}{target_atom}{i}/init_2/control.in"

        # Change control file
        with open(control, "r") as read_control:
            control_content = read_control.readlines()

        # Replace specific lines
        for j, line in enumerate(control_content):
            spl = line.split()

            if len(spl) > 1:
                # Fix basis sets
                if "species" == spl[0] and target_atom == spl[1]:
                    if found_target_atom is False:
                        control_content[j] = f"  species        {target_atom}1\n"
                        found_target_atom = True

                # Change keyword lines
                if "sc_iter_limit" in spl:
                    control_content[j] = iter_limit
                if "restart_write_only" in spl:
                    control_content[j] = restart_file
                if "force_single_restartfile" in spl:
                    control_content[j] = restart_force
                if "#force_occupation_projector" == spl[0]:
                    control_content[j] = fop
                if "charge" in spl:
                    control_content[j] = charge
                if "#" == spl[0] and "charge" == spl[1]:
                    control_content[j] = charge

        # Check if parameters not found
        no_iter_limit = False
        no_restart = False
        no_charge = False

        if iter_limit not in control_content:
            no_iter_limit = True
        if restart_file not in control_content:
            no_restart = True
        if charge not in control_content:
            no_charge = True

        # Write the data to the file
        with open(control, "w+") as write_control:
            write_control.writelines(control_content)

            # Append parameters to end of file if not found
            if no_iter_limit is True:
                write_control.write(iter_limit)
            if no_restart is True:
                write_control.write(restart_file)
            if no_charge is True:
                write_control.write(charge)

        # Add 0.1 charge
        with open(control, "r") as read_control:
            control_content = read_control.readlines()

        # Replace specific lines
        for j, line in enumerate(control_content):
            spl = line.split()

            if target_atom + "1" in spl:
                # Add to nucleus
                if f"    nucleus             {at_num}\n" in control_content[j:]:
                    # nucleus = control_content[n_index]  # save for hole
                    control_content[n_index] = f"    nucleus             {at_num}.1\n"
                elif f"    nucleus      {at_num}\n" in control_content[j:]:
                    # nucleus = control_content[n_index]  # save for hole
                    control_content[n_index] = f"    nucleus      {at_num}.1\n"

                # Add to valence orbital
                if "#     ion occupancy\n" in control_content[j:]:

                    # Add the 0.1 electron
                    control_content[valence_index] = atom_valence
                    break

        with open(control, "w+") as write_control:
            write_control.writelines(control_content)

    print("init_2 files written successfully")


def setup_hole(
    calc_path,
    ks_states,
    target_atom,
    num_atom,
    nucleus,
    valence,
    n_index,
    valence_index,
):
    """Write new hole directories and control files to calculate FOP."""

    # occ_type = 'occupation_type         gaussian 0.1\n'
    # iter_limit = 'sc_iter_limit             20\n'
    init_iter = "sc_init_iter              75\n"
    ks_method = "KS_method                serial\n"
    # mixer = 'mixer                    pulay\n'
    # charge_mix = 'charge_mix_param          0.02\n'
    restart = "restart_read_only       restart_file\n"
    charge = "charge                    1.0\n"
    fop = f"force_occupation_projector {ks_states[0]} 1 0.0 {ks_states[0]} {ks_states[1]}\n"
    output_cube = "output                  cube spin_density\n"
    output_mull = "#output                  mulliken\n"
    output_hirsh = "#output                  hirshfeld\n"

    # Calculate original valence state
    val_spl = valence.split(".")
    del val_spl[-1]
    val_spl.append(".\n")
    valence = "".join(val_spl)

    if type(num_atom) == list:
        loop_iterator = num_atom
    else:
        loop_iterator = range(num_atom)

    for i in loop_iterator:
        if type(num_atom) != list:
            i += 1

        os.makedirs(f"{calc_path}{target_atom}{i}/hole", exist_ok=True)
        shutil.copyfile(
            f"{calc_path}{target_atom}{i}/init_1/geometry.in",
            f"{calc_path}{target_atom}{i}/hole/geometry.in",
        )
        shutil.copyfile(
            f"{calc_path}{target_atom}{i}/init_1/control.in",
            f"{calc_path}{target_atom}{i}/hole/control.in",
        )

        control = f"{calc_path}{target_atom}{i}/hole/control.in"

        with open(control, "r") as read_control:
            control_content = read_control.readlines()

        # Set nuclear and valence orbitals back to integer values
        control_content[n_index] = nucleus
        control_content[valence_index] = valence

        # Replace specific lines
        for j, line in enumerate(control_content):
            spl = line.split()

            if len(spl) > 1:
                # Change keyword lines
                # if 'occupation_type' in spl:
                #     control_content[j] = occ_type
                # if 'sc_iter_limit' in spl:
                #     control_content[j] = iter_limit
                if "#sc_init_iter" in spl:
                    control_content[j] = init_iter
                if "#" == spl[0] and "sc_init_iter" == spl[1]:
                    control_content[j] = init_iter
                if "KS_method" in spl:
                    control_content[j] = ks_method
                # if 'mixer' in spl:
                #     control_content[j] = mixer
                if "restart" in spl or "restart_write_only" in spl:
                    control_content[j] = restart
                if "#force_occupation_projector" == spl[0]:
                    control_content[j] = fop
                if "#" == spl[0] and "force_occupation_projector" == spl[1]:
                    control_content[j] = fop
                if "charge" in spl:
                    control_content[j] = charge
                # if 'charge_mix_param' in spl:
                #     control_content[j] = charge_mix
                if ["#output", "cube", "spin_density"] == spl or [
                    "#",
                    "output",
                    "cube",
                    "spin_density",
                ] == spl:
                    control_content[j] = output_cube
                if ["#output", "hirshfeld"] == spl or [
                    "#",
                    "output",
                    "hirshfeld",
                ] == spl:
                    control_content[j] = output_hirsh
                if ["#output", "mulliken"] == spl or [
                    "#",
                    "output",
                    "mulliken",
                ] == spl:
                    control_content[j] = output_mull

        # Check if parameters not found
        # no_occ_type = False
        no_init_iter = False
        # no_iter_limit = False
        # no_mixer = False
        no_restart = False
        no_fop = False
        no_charge = False
        # no_charge_mix = False
        no_output_cube = False
        # no_output_mull = False
        # no_output_hirsh = False

        # TODO finish adding mixer stuff
        # if occ_type not in control_content:
        #     no_occ_type = True
        # if iter_limit not in control_content:
        #     no_iter_limit = True
        if init_iter not in control_content:
            no_init_iter = True
        # if mixer not in control_content:
        #     no_mixer = True
        if restart not in control_content:
            no_restart = True
        if fop not in control_content:
            no_fop = True
        if charge not in control_content:
            no_charge = True
        # if charge_mix not in control_content:
        #     no_charge_mix = True
        if output_cube not in control_content:
            no_output_cube = True
        # if output_mull not in control_content:
        #     no_output_mull = True
        # if output_hirsh not in control_content:
        #     no_output_hirsh = True

        # Write the data to the file
        with open(control, "w+") as write_control:
            write_control.writelines(control_content)

            # Append parameters to end of file if not found
            # if no_occ_type is True:
            #     write_control.write(occ_type)
            # if no_iter_limit is True:
            #     write_control.write(iter_limit)
            if no_init_iter is True:
                write_control.write(init_iter)
            # if no_mixer is True:
            #     write_control.write(mixer)
            if no_restart is True:
                write_control.write(restart)
            if no_fop is True:
                write_control.write(fop)
            if no_charge is True:
                write_control.write(charge)
            # if no_charge_mix is True:
            #     write_control.write(charge_mix)
            if no_output_cube is True:
                write_control.write(output_cube)
            # if no_output_mull is True:
            #     write_control.write(output_mull)
            # if no_output_hirsh is True:
            #     write_control.write(output_hirsh)

    print("hole files written successfully")


def setup_fob(target_atom, num_atom, ks_max, occ_type, run_loc, ad_cont_opts):
    """Write new directories and control files to calculate FOB."""

    # TODO allow greater control over which atoms to constrain
    # eg. see isopropanol

    # The new basis method should utilise ks method parallel
    ks_method = ""
    if occ_type == "force_occupation_basis":
        ks_method = "KS_method               serial\n"
    if occ_type == "deltascf_basis":
        ks_method = "KS_method               parallel\n"

    charge = "charge                  1.0\n"
    cube = "output                  cube spin_density\n"

    for i in range(num_atom):
        i += 1
        os.makedirs(f"{run_loc}/{target_atom}{i}/hole/", exist_ok=True)
        shutil.copyfile(
            f"{run_loc}/ground/control.in",
            f"{run_loc}/{target_atom}{i}/hole/control.in",
        )
        shutil.copyfile(
            f"{run_loc}/ground/geometry.in",
            f"{run_loc}/{target_atom}{i}/hole/geometry.in",
        )

        control = f"{run_loc}/{target_atom}{i}/hole/control.in"

        fob = ""
        if occ_type == "force_occupation_basis":
            fob = f"{occ_type}  {i} 1 atomic 2 1 1 0.0 {ks_max}\n"
        elif occ_type == "deltascf_basis":
            fob = f"{occ_type}          {i} 1 atomic 2 1 1 0.0 {ks_max}\n"

        # Find and replace stuff to be changed
        with open(control, "r") as read_control:
            content = read_control.readlines()

        # Replace specific lines
        for j, line in enumerate(content):
            spl = line.split()

            # Some error checking
            if len(spl) > 1:
                if "force_occupation_basis" == spl[0]:
                    print("force_occupation_basis keyword already found in control.in")
                    exit(1)
                if "charge" == spl[0]:
                    print("charge keyword already found in control.in")
                    exit(1)
                if "output" == spl[0] and "cube" == spl[1] and "spin_density" == spl[2]:
                    print("spin_density cube output already specified in control.in")

                # Change keyword lines
                if "KS_method" in spl:
                    content[j] = ks_method
                if "#force_occupation_basis" in spl:
                    content[j] = fob
                if "#" == spl[0] and "force_occupation_basis" == spl[1]:
                    content[j] = fob
                if "#charge" in spl:
                    content[j] = charge
                if "#" == spl[0] and "charge" == spl[1]:
                    content[j] = charge
                if line.strip() == "#output                  cube spin_density":
                    content[j] = cube
                if "#" == spl[0] and "output" == spl[1]:
                    content[j] = cube

        # Check if parameters not found
        no_ks = False
        no_fob = False
        no_charge = False
        no_cube = False

        if ks_method not in content:
            no_ks = True
        if fob not in content:
            no_fob = True
        if charge not in content:
            no_charge = True
        if cube not in content:
            no_cube = True

        # Write the data to the file
        with open(control, "w+") as write_control:
            write_control.writelines(content)

            # Append parameters to end of file if not found
            if no_ks is True:
                write_control.write(ks_method)
            if no_fob is True:
                write_control.write(fob)
            if no_charge is True:
                write_control.write(charge)
            if no_cube is True:
                write_control.write(cube)

            # Append additional parameters specified by the user to control.in
            for opt in ad_cont_opts:
                write_control.write(opt)

    print("Files and directories written successfully")
