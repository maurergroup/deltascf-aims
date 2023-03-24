"""Automate creation of files for FOP calculations in FHI-aims."""

import glob
import os
import shutil
import subprocess

import numpy as np
import yaml


class ForceOccupation:
    """Manipulate FHIaims input files to setup basis and projector calculations."""

    def __init__(
        self,
        constr_atoms,
        spec_at_constr,
        element_symbols,
        run_loc,
        geometry,
        control,
        ad_cont_opts,
    ):
        self.constr_atoms = constr_atoms
        self.spec_at_constr = spec_at_constr
        self.element_symbols = element_symbols
        self.run_loc = run_loc
        self.geometry = geometry
        self.control = control
        self.ad_cont_opts = ad_cont_opts

        self.atom_specifier = []

        # All supported elements
        with open("elements.yml", "r") as elements:
            self.elements = yaml.load(elements, Loader=yaml.SafeLoader)

    def read_ground_inp(self, geometry_path):
        """Find the number of atoms in the geometry file."""

        # For if the user supplied element symbols to constrain
        if self.constr_atoms is not None:
            # Check validity of specified elements
            for atom in self.constr_atoms:
                if atom not in self.elements:
                    raise ValueError("invalid element specified")

            print(
                "atoms argument not specified, defaulting to all target atoms in geometry.in"
            )

            # Constrain all atoms of the target element
            for atom in self.constr_atoms:
                with open(geometry_path, "r") as geom_in:
                    atom_counter = 0

                    for line in geom_in:
                        spl = line.split()

                        if len(spl) > 0 and "atom" == spl[0]:
                            atom_counter += 1
                            element = spl[-1]  # Identify atom
                            identifier = spl[0]  # Extra check that line is an atom

                            if identifier == "atom" and element == atom:
                                self.atom_specifier.append(atom_counter)

        # For if the user supplied atom indices to constrain
        if self.element_symbols is not None:
            # Check validity of specified elements
            for atom in self.element_symbols:
                if atom not in self.elements:
                    raise ValueError("invalid element specified")

            self.atom_specifier = self.spec_at_constr

        print("specified atoms:", self.atom_specifier)

        return self.atom_specifier

    def get_electronic_structure(self, atom):
        """Get valence electronic structure of target atom."""
        # Adapted from scipython.com question P2.5.12

        self.atom_index = self.elements[str(atom)]["number"]

        # Letters identifying subshells
        l_letter = ["s", "p", "d", "f", "g"]

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
        s_config = ""

        for i in range(len(self.elements[: self.atom_index])):
            nelec += 1

            # this subshell is now full
            if nelec > 2 * (2 * l + 1):
                # The most recent configuration was for a Noble gas: store it
                if l == 1:
                    # Turn a list of orbital, nelec pairs into a configuration string
                    noble_gas_config = (
                        ".".join(["{:2s}{:d}".format(*e) for e in config]),
                        "[{}]".format(self.elements[i - 1]),
                    )

                # Start a new subshell
                inl += 1
                _, n, l = nl_pairs[inl]
                config.append(["{}{}".format(n, l_letter[l]), 1])
                nelec = 1

            # add an electron to the current subshell
            else:
                config[-1][1] += 1

            # Turn config into a string
            s_config = ".".join(["{:2s}{:d}".format(*e) for e in config])

            # Replace the heaviest Noble gas configuration with its symbol
            s_config = s_config.replace(*noble_gas_config)

        output = list(s_config.split(".").pop(-1))
        self.valence = f"    valence      {output[0]}  {output[1]}   {output[2]}.1\n"

        return self.atom_index, self.valence

    @staticmethod
    def mod_keywords(ad_cont_opts, opts):
        """Allow users to modify and add keywords"""

        for ad_opt in list(ad_cont_opts):
            spl_key = ad_opt.split(" ", 1)[0]

            # Split the keyword and its value
            try:
                spl_val = ad_opt.split(" ", 1)[1]
            except IndexError:
                spl_val = None

            # Check if the keyword is already in the input file
            for opt in list(opts):
                if spl_key == opt:
                    if spl_val is not None:
                        opts[opt] = spl_val
                    else:
                        del opts[opt]

            # Add new keywords
            if spl_key not in list(opts):
                if spl_val is not None:
                    opts.update({spl_key: spl_val})
                else:
                    opts.update({spl_key: ""})

        return opts

    @staticmethod
    def change_control_keywords(control, opts):
        """Modify the keywords in a control.in file from a dictionary of options."""

        # Find and replace keywords in control file
        with open(control, "r") as read_control:
            content = read_control.readlines()

        divider = "#==============================================================================="

        # Change keyword lines
        for opt in opts:
            ident = 0

            for j, line in enumerate(content):
                spl = line.split()

                if opt in spl:
                    content[j] = f"{opt:<35} {opts[opt]}\n"
                    break

                # Marker if ASE was used so keywords will be added in the same place as the others
                elif divider in line:
                    ident += 1
                    if ident == 3:
                        content.insert(j, f"{opt:<35} {opts[opt]}\n")
                        break

            # For when a non-ASE input file is used
            if ident == 0:
                content.append(f"{opt:<35} {opts[opt]}\n")

        return content

    @staticmethod
    def add_partial_charge(content, target_atom, at_num, atom_valence, charge):
        """Add a partial charge to a basis set in a control.in file."""

        # Ensure returned variables are bound
        n_index = 0
        valence_index = 0
        nucleus = None

        for j, line in enumerate(content):
            spl = line.split()

            if target_atom + "1" in spl:
                # Add to nucleus
                if f"    nucleus             {at_num}\n" in content[j:]:
                    n_index = (
                        content[j:].index(f"    nucleus             {at_num}\n") + j
                    )
                    nucleus = content[n_index]  # save for hole
                    content[n_index] = f"    nucleus             {at_num + charge}\n"
                elif f"    nucleus      {at_num}\n" in content[j:]:
                    n_index = content[j:].index(f"    nucleus      {at_num}\n") + j
                    nucleus = content[n_index]  # save for hole
                    content[n_index] = f"    nucleus      {at_num + charge}\n"

                # Add to valence orbital
                if "#     ion occupancy\n" in content[j:]:
                    vbs_index = content[j:].index("#     valence basis states\n") + j
                    io_index = content[j:].index("#     ion occupancy\n") + j

                    # Check which orbital to add 0.1 to
                    principle_qns = np.array([])
                    azimuthal_orbs = np.array([])
                    azimuthal_qns = np.zeros(io_index - vbs_index - 1)
                    azimuthal_refs = {"s": 1, "p": 2, "d": 3, "f": 4, "g": 5}

                    # Get azimuthal and principle quantum numbers
                    for count, valence_orbital in enumerate(
                        content[vbs_index + 1 : io_index]
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
                    # valence = content[
                    #     vbs_index + addition_state + 1
                    # ]  # save for write hole file
                    valence_index = vbs_index + addition_state + 1
                    content[valence_index] = atom_valence
                    break

        return n_index, valence_index, nucleus, content


class Projector(ForceOccupation):
    """Create input files for projector calculations."""

    def __init__(self, parent_instance):
        """Inherit all the variables from an instance of the parent class"""
        vars(self).update(vars(parent_instance))

    def setup_init_1(
        self,
        basis_set,
        defaults,
    ):
        """Write new directories and control files for the first initialisation calculation."""

        # Default control file options
        opts = {
            "xc": "pbe",
            "spin": "collinear",
            "default_initial_moment": 0,
            "charge": 0.1,
            "restart_write_only": "restart_file",
            "restart_save_iterations": 20,
        }

        # Ensure returned variables are bound
        # n_index = 0
        # valence_index = 0
        # nucleus = None

        self.mod_keywords(self.ad_cont_opts, opts)

        # Create a new intermediate file and write basis sets to it
        shutil.copyfile(self.control, f"{self.run_loc}ground/control.in.new")

        # Find species defaults location from location of binary
        for el in self.constr_atoms:
            basis_set = glob.glob(f"{defaults}/defaults_2020/{basis_set}/*{el}_default")
            bash_add_basis = f"cat {basis_set[0]}"
            new_control = open(f"{self.run_loc}ground/control.in.new", "a")
            subprocess.run(bash_add_basis.split(), check=True, stdout=new_control)

            for i in range(len(self.atom_specifier)):
                i += 1

                os.makedirs(f"{self.run_loc}{el}{i}/init_1", exist_ok=True)
                shutil.copyfile(
                    f"{self.run_loc}ground/control.in.new",
                    f"{self.run_loc}{el}{i}/init_1/control.in",
                )
                shutil.copyfile(
                    f"{self.run_loc}ground/geometry.in",
                    f"{self.run_loc}{el}{i}/init_1/geometry.in",
                )

                # Change geometry file
                with open(self.geometry, "r") as read_geom:
                    geom_content = read_geom.readlines()

                # Change atom to {atom}{num}
                atom_counter = 0
                for j, line in enumerate(geom_content):
                    spl = line.split()

                    if "atom" in line and el in line:
                        if atom_counter + 1 == i:
                            partial_hole_atom = f" {el}1\n"
                            geom_content[j] = " ".join(spl[0:-1]) + partial_hole_atom

                        atom_counter += 1

                with open(self.geometry, "w") as write_geom:
                    write_geom.writelines(geom_content)

                # Change control file
                control_content = self.change_control_keywords(self.control, opts)
                (
                    self.n_index,
                    self.valence_index,
                    self.nucleus,
                    self.control_content,
                ) = self.add_partial_charge(
                    control_content,
                    el,
                    self.elements[el]["number"],
                    self.valence,
                    opts["charge"],
                )

                with open(self.control, "w") as write_control:
                    write_control.writelines(control_content)

        print("init_1 files written successfully")

    def setup_init_2(
        self,
        ks_start,
        ks_stop,
        occ,
        spin,
        calc_path,
        target_atom,
        num_atom,
        at_num,
        atom_valence,
        n_index,
        valence_index,
        occ_type,
    ):
        """Write new directories and control files for the second initialisation calculation."""

        opts = {
            "xc": "pbe",
            "spin": "collinear",
            "default_initial_moment": 0,
            "charge": 1.1,
            "sc_iter_limit": 1,
            occ_type: f"{atom_index} {spin} {occ}, {ks_start} {ks_stop}",
            "restart": "restart_file",
            "restart_save_iterations": 20,
        }

        self.mod_keywords(self.ad_cont_opts, opts)

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
                        control_content[
                            n_index
                        ] = f"    nucleus             {at_num}.1\n"
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

    @staticmethod
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


class Basis(ForceOccupation):
    """Create input files for basis calculations."""

    def setup_basis(
        self, target_atom, num_atom, occ_no, ks_max, occ_type, run_loc, ad_cont_opts
    ):
        """Write new directories and control files for basis calculations."""

        # The new basis method should utilise ks method parallel
        ks_method = ""
        if occ_type == "force_occupation_basis":
            ks_method = "serial"
        if occ_type == "deltascf_basis":
            ks_method = "parallel"

        # Default control file options
        opts = {
            "xc": "pbe",
            "spin": "collinear",
            "default_initial_moment": 0,
            "charge": 1.0,
            occ_type: f"1 1 atomic 2 1 1 {occ_no} {ks_max}",
            "KS_method": ks_method,
            # "output": "cube spin_density",
        }

        # Allow users to modify and add keywords
        opts = self.mod_keywords(ad_cont_opts, opts)

        # Iterate over each constrained atom
        for i in range(num_atom):
            i += 1

            # Create new directories and .in files for each constrained atom
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

            # Change control file
            content = self.change_control_keywords(control, opts)

            # Write the data to the file
            with open(control, "w") as write_control:
                write_control.writelines(content)

        print("Files and directories written successfully")
