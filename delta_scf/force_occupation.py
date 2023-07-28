"""Automate creation of files for FOP calculations in FHI-aims."""

import glob
import os
import shutil
import subprocess

import numpy as np
import yaml


class ForceOccupation:
    """Manipulate FHIaims input files to setup basis and projector calculations."""

    def __init__(self, element_symbols, run_loc, geometry, ad_cont_opts, species):
        self.element_symbols = element_symbols
        self.run_loc = run_loc
        self.geometry = geometry
        self.ad_cont_opts = ad_cont_opts
        self.species = species

        # Convert k_grid key to a string from a tuple
        # Writing the options for a hole calculation doesn't use ASE, so it must be
        # converted to a string here
        if "k_grid" in ad_cont_opts.keys():
            ad_cont_opts["k_grid"] = " ".join(map(str, ad_cont_opts["k_grid"]))

        self.new_control = f"{self.run_loc}/ground/control.in.new"
        self.atom_specifier = []

        # Find the root directory of the package
        self.current_path = os.path.dirname(os.path.realpath(__file__))

        # All supported elements
        with open(f"{self.current_path}/elements.yml", "r") as elements:
            self.elements = yaml.load(elements, Loader=yaml.SafeLoader)

    def read_ground_inp(self, constr_atoms, spec_at_constr, geometry_path):
        """Find the number of atoms in the geometry file."""

        # For if the user supplied element symbols to constrain
        if constr_atoms is not None:
            # Check validity of specified elements
            for atom in constr_atoms:
                if atom not in self.elements:
                    raise ValueError("invalid element specified")

            print(
                "atoms argument not specified, defaulting to all target atoms in geometry.in"
            )

            # Constrain all atoms of the target element
            for atom in constr_atoms:
                with open(geometry_path, "r") as geom_in:
                    atom_counter = 0

                    for line in geom_in:
                        spl = line.split()

                        if len(spl) > 0 and "atom" in spl[0]:
                            atom_counter += 1
                            element = spl[-1]  # Identify atom
                            identifier = spl[0]  # Extra check that line is an atom

                            if "atom" in identifier and element == atom:
                                self.atom_specifier.append(atom_counter)

        # For if the user supplied atom indices to constrain
        if len(spec_at_constr) > 0:
            # Check validity of specified elements
            for atom in self.element_symbols:
                if atom not in self.elements:
                    raise ValueError("invalid element specified")

            self.atom_specifier = list(spec_at_constr)

        print("specified atom indices:", self.atom_specifier)

        return self.atom_specifier

    def get_electronic_structure(self, atom):
        """Get valence electronic structure of target atom."""
        # Adapted from scipython.com question P2.5.12

        # Get the atomic number
        self.atom_index = self.elements.index(str(atom)) + 1

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

        # Correct transition metal atoms where the s orbital is filled before d
        if output[1] == "d":
            if output[2] == "4" or output[2] == "9":
                output[0] = str(int(output[0]) + 1)
                output[1] = "s"
                output[2] = "1"

        self.valence = f"    valence      {output[0]}  {output[1]}   {output[2]}.1\n"

    @staticmethod
    def add_additional_basis(current_path, elements, content, target_atom):
        """Add an additional basis set for the core hole calculation."""

        # Check the additional functions haven't already been added to control
        for line in content:
            if "# Additional basis functions for atom with a core hole" in line:
                return

        # Get the additional basis set
        if "utils" in current_path.split("/"):
            with open(f"{current_path}/../delta_scf/add_basis_functions.yml", "r") as f:
                ad_basis = yaml.safe_load(f)
        else:
            with open(f"{current_path}/add_basis_functions.yml", "r") as f:
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
            pass

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
        separator = (
            "#######################################################################"
            "#########"
        )
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

        else:
            print(
                "Warning: there was an error with adding the additional basis functions"
            )
            return content

    @staticmethod
    def get_control_keywords(control):
        """Get the keywords in a control.in file"""

        # Find and replace keywords in control file
        with open(control, "r") as read_control:
            content = read_control.readlines()

        # Get keywords
        opts = {}
        for line in content:
            spl = line.split()

            # Break when basis set definitions start
            if (
                "############################################################"
                "####################" in line
            ):
                break

            # Add the dictionary value as a string
            if len(spl) > 1 and "#" not in spl[0]:
                if len(spl[1:]) > 1:
                    opts[spl[0]] = " ".join(spl[1:])

                else:
                    opts[spl[0]] = spl[1]

        return opts

    @staticmethod
    def mod_keywords(ad_cont_opts, opts):
        """Allow users to modify and add keywords"""

        for key in list(ad_cont_opts.keys()):
            opts.update({key: ad_cont_opts[key]})

        return opts

    @staticmethod
    def change_control_keywords(control, opts):
        """Modify the keywords in a control.in file from a dictionary of options."""

        # Find and replace keywords in control file
        with open(control, "r") as read_control:
            content = read_control.readlines()

        divider_1 = "#==============================================================================="
        divider_2 = "################################################################################"

        # Change keyword lines
        for opt in opts:
            ident = 0

            for j, line in enumerate(content):
                spl = line.split()

                # print(opts)
                # print(opts[opt])
                # print(opt)

                if opt in spl:
                    content[j] = f"{opt:<34} {opts[opt]}\n"
                    break

                # Marker if ASE was used so keywords will be added in the same
                # place as the others
                elif divider_1 in line:
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
            # if ident == 0:
            #     content.insert(0, f"{opt:<34} {opts[opt]}\n")

        return content

    @staticmethod
    def add_partial_charge(content, target_atom, at_num, atom_valence, partial_charge):
        """Add a partial charge to a basis set in a control.in file."""

        # Ensure returned variables are bound
        nuclear_index = 0
        valence_index = 0
        nucleus = ""

        for j, line in enumerate(content):
            spl = line.split()

            if target_atom + "1" in spl:
                # Add to nucleus
                if f"    nucleus             {at_num}\n" in content[j:]:
                    nuclear_index = (
                        content[j:].index(f"    nucleus             {at_num}\n") + j
                    )
                    nucleus = content[nuclear_index]  # save for hole
                    content[
                        nuclear_index
                    ] = f"    nucleus             {at_num + partial_charge}\n"
                elif f"    nucleus      {at_num}\n" in content[j:]:
                    nuclear_index = (
                        content[j:].index(f"    nucleus      {at_num}\n") + j
                    )
                    nucleus = content[nuclear_index]  # save for hole
                    content[
                        nuclear_index
                    ] = f"    nucleus      {at_num + partial_charge}\n"

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
                    highest_nuclear_index = np.where(principle_qns == highest_n)

                    # Check for highest l if 2 orbitals have the same n
                    if len(highest_nuclear_index[0]) > 1:
                        highest_l = np.amax(azimuthal_qns)
                        highest_l_index = np.where(azimuthal_qns == highest_l)
                        addition_state = np.intersect1d(
                            highest_nuclear_index, highest_l_index
                        )[0]
                    else:
                        addition_state = highest_nuclear_index[0][0]

                    # Add the 0.1 electron
                    valence_index = vbs_index + addition_state + 1
                    content[valence_index] = atom_valence
                    break

        return nuclear_index, valence_index, nucleus, content


class Projector(ForceOccupation):
    """Create input files for projector calculations."""

    def __init__(self, parent_instance):
        """Inherit all the variables from an instance of the parent class"""
        vars(self).update(vars(parent_instance))

    def setup_init_1(self, basis_set, defaults, control):
        """Write new directories and control files for the first initialisation calculation."""

        # Default control file options
        opts = {
            # "xc": "pbe",
            # "spin": "collinear",
            # "default_initial_moment": 0,
            "charge": 0.1,
            "sc_iter_limit": 500,
            "sc_init_iter": 75,
            "restart_write_only": "restart_file",
            "restart_save_iterations": 5,
        }

        # Add or change user-specified keywords to the control file
        opts = self.mod_keywords(self.ad_cont_opts, opts)

        # Create a new intermediate file and write basis sets to it
        shutil.copyfile(control, self.new_control)

        # Loop over each element to constrain
        for el in self.element_symbols:
            # Find species defaults location from location of binary
            basis_set = glob.glob(f"{defaults}/defaults_2020/{basis_set}/*{el}_default")
            bash_add_basis = f"cat {basis_set[0]}"

            # Create a new intermediate control file
            new_control = open(self.new_control, "a")
            subprocess.run(bash_add_basis.split(), check=True, stdout=new_control)

            # Change basis set label for core hole atom
            with open(self.new_control, "r") as read_control:
                new_basis_content = read_control.readlines()

            found_target_atom = False

            for j, line in enumerate(new_basis_content):
                spl = line.split()

                if len(spl) > 1:
                    if "species" == spl[0] and el == spl[1]:
                        if found_target_atom is False:
                            new_basis_content[j] = f"  species        {el}1\n"
                            found_target_atom = True

            # Write it to intermediate control file
            with open(self.new_control, "w") as write_control:
                write_control.writelines(new_basis_content)

            # Loop over each individual atom to constrain
            for i in self.atom_specifier:
                # TODO: fix this for individual atom constraints

                i1_control = f"{self.run_loc}/{el}{i}/init_1/control.in"
                i1_geometry = f"{self.run_loc}/{el}{i}/init_1/geometry.in"

                # Create new directory and control file for init_1 calc
                os.makedirs(f"{self.run_loc}/{el}{i}/init_1", exist_ok=True)
                shutil.copyfile(
                    self.new_control,
                    i1_control,
                )
                # Create new geometry file for init_1 calc
                shutil.copyfile(f"{self.run_loc}/ground/geometry.in", i1_geometry)

                # Change geometry file
                with open(i1_geometry, "r") as read_geom:
                    geom_content = read_geom.readlines()

                # Change core hole atom to {atom}{num}
                atom_counter = 0
                for j, line in enumerate(geom_content):
                    spl = line.split()

                    if "atom" in line and el in line:
                        if atom_counter + 1 == i:
                            partial_hole_atom = f" {el}1\n"
                            geom_content[j] = " ".join(spl[0:-1]) + partial_hole_atom

                        atom_counter += 1

                with open(i1_geometry, "w") as write_geom:
                    write_geom.writelines(geom_content)

                # Change control file
                control_content = self.change_control_keywords(i1_control, opts)

                # Add additional core-hole basis functions
                control_content = self.add_additional_basis(
                    self.current_path, self.elements, control_content, f"{el}1"
                )

                (
                    self.n_index,
                    self.v_index,
                    self.nucleus,
                    control_content,
                ) = self.add_partial_charge(
                    control_content,
                    el,
                    self.elements.index(el) + 1,
                    self.valence,
                    opts["charge"],
                )

                with open(i1_control, "w") as write_control:
                    write_control.writelines(control_content)

        print("init_1 files written successfully")

    def setup_init_2(self, ks_start, ks_stop, occ, occ_type, spin, pbc):
        """Write new directories and control files for the second initialisation calculation."""

        ks_method = ""
        if occ_type == "force_occupation_projector":
            ks_method = "serial"
        if occ_type == "deltascf_projector" and pbc is False:
            ks_method = "parallel"
        if occ_type == "deltascf_projector" and pbc is True:
            ks_method = "serial"

        # Loop over each element to constrain
        for el in self.element_symbols:
            # Loop over each individual atom to constrain
            for i in self.atom_specifier:
                opts = {
                    # "xc": "pbe",
                    # "spin": "collinear",
                    # "default_initial_moment": 0,
                    "charge": 1.1,
                    "sc_iter_limit": 1,
                    occ_type: f"{ks_start} {spin} {occ} {ks_start} {ks_stop}",
                    "KS_method": ks_method,
                    "restart": "restart_file",
                    "restart_save_iterations": 1,
                }

                # Add or change user-specified keywords to the control file
                opts = self.mod_keywords(self.ad_cont_opts, opts)

                i2_control = f"{self.run_loc}/{el}{i}/init_2/control.in"

                # Create new directory for init_2 calc
                os.makedirs(f"{self.run_loc}/{el}{i}/init_2", exist_ok=True)
                shutil.copyfile(self.new_control, i2_control)
                shutil.copyfile(
                    f"{self.run_loc}/{el}{i}/init_1/geometry.in",
                    f"{self.run_loc}/{el}{i}/init_2/geometry.in",
                )

                # Change control file
                control_content = self.change_control_keywords(i2_control, opts)

                # Add additional core-hole basis functions
                control_content = self.add_additional_basis(
                    self.current_path, self.elements, control_content, f"{el}1"
                )

                # Add partial charge to the control file
                _, _, _, control_content = self.add_partial_charge(
                    control_content,
                    el,
                    self.elements.index(el) + 1,
                    self.valence,
                    opts["charge"] - 1,
                )

                with open(i2_control, "w") as write_control:
                    write_control.writelines(control_content)

        print("init_2 files written successfully")

    def setup_hole(self, ks_start, ks_stop, occ, occ_type, spin, pbc):
        """Write new hole directories and control files for the hole calculation."""

        # Calculate original valence state
        val_spl = self.valence.split(".")
        del val_spl[-1]
        val_spl.append(".\n")
        self.valence = "".join(val_spl)

        # The new basis method should utilise ks method parallel
        ks_method = ""
        if occ_type == "force_occupation_projector":
            ks_method = "serial"
        if occ_type == "deltascf_projector" and pbc is False:
            ks_method = "parallel"
        if occ_type == "deltascf_projector" and pbc is True:
            ks_method = "serial"

        # Loop over each element to constrain
        for el in self.element_symbols:
            # Loop over each individual atom to constrain
            for i in self.atom_specifier:
                opts = {
                    # "xc": "pbe",
                    # "spin": "collinear",
                    # "default_initial_moment": 0,
                    "charge": 1.0,
                    "sc_iter_limit": 500,
                    "sc_init_iter": 75,
                    occ_type: f"{ks_start} {spin} {occ} {ks_start} {ks_stop}",
                    "KS_method": ks_method,
                    "restart_read_only": "restart_file",
                    "output": "cube spin_density",
                }

                # Add or change user-specified keywords to the control file
                opts = self.mod_keywords(self.ad_cont_opts, opts)

                # Location of the hole control file
                h_control = f"{self.run_loc}/{el}{i}/hole/control.in"

                # Create new directory for hole calc
                os.makedirs(f"{self.run_loc}/{el}{i}/hole", exist_ok=True)
                shutil.copyfile(
                    f"{self.run_loc}/{el}{i}/init_1/geometry.in",
                    f"{self.run_loc}/{el}{i}/hole/geometry.in",
                )
                shutil.copyfile(self.new_control, h_control)

                with open(h_control, "r") as read_control:
                    control_content = read_control.readlines()

                # Set nuclear and valence orbitals back to integer values
                control_content[self.n_index] = self.nucleus
                control_content[self.v_index] = self.valence

                # Change control file
                control_content = self.change_control_keywords(h_control, opts)

                # Add additional core-hole basis functions
                control_content = self.add_additional_basis(
                    self.current_path, self.elements, control_content, f"{el}1"
                )

                # Remove partial charge from the control file
                _, _, _, control_content = self.add_partial_charge(
                    control_content, el, self.elements.index(el) + 1, self.valence, 0
                )

                # Write the data to the file
                with open(h_control, "w") as write_control:
                    write_control.writelines(control_content)

        print("hole files written successfully")


class Basis(ForceOccupation):
    """Create input files for basis calculations."""

    def __init__(self, parent_instance):
        """Inherit all the variables from an instance of the parent class"""
        vars(self).update(vars(parent_instance))

    def setup_basis(self, spin, n_qn, l_qn, m_qn, occ_no, ks_max, occ_type):
        """Write new directories and control files for basis calculations."""

        # The new basis method should utilise ks method parallel
        ks_method = ""
        if occ_type == "force_occupation_basis":
            ks_method = "serial"
        if occ_type == "deltascf_basis":
            ks_method = "parallel"

        # Iterate over each constrained atom
        for el in self.element_symbols:
            # Loop over each individual atom to constrain
            for i in range(len(self.atom_specifier)):
                # Default control file options
                opts = {
                    # "xc": "pbe",
                    # "spin": "collinear",
                    # "default_initial_moment": 0,
                    "charge": 1.0,
                    "sc_iter_limit": 500,
                    "sc_init_iter": 75,
                    occ_type: f"{self.atom_specifier[i]} {spin} atomic {n_qn} {l_qn} {m_qn} {occ_no} {ks_max}",
                    "KS_method": ks_method,
                }

                # Allow users to modify and add keywords
                opts = self.mod_keywords(self.ad_cont_opts, opts)

                i += 1

                control = f"{self.run_loc}/{el}{i}/control.in"

                # Create new directories and .in files for each constrained atom
                os.makedirs(f"{self.run_loc}/{el}{i}/", exist_ok=True)
                shutil.copyfile(
                    f"{self.run_loc}/ground/control.in",
                    control,
                )
                shutil.copyfile(
                    f"{self.run_loc}/ground/geometry.in",
                    f"{self.run_loc}/{el}{i}/geometry.in",
                )

                # Change control file
                content = self.change_control_keywords(control, opts)

                # Write the data to the file
                with open(control, "w") as write_control:
                    write_control.writelines(content)

        print("files and directories written successfully")
