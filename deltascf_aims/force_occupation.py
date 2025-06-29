import glob
import os
import shutil
import warnings

import yaml

from deltascf_aims.utils import utils


class ForceOccupation:
    """
    Manipulate FHIaims input files to setup basis and projector calculations.

    ...

    Attributes
    ----------
        element_symbols : List[str]
            list of element symbols to constrain


    TODO
    """

    def __init__(
        self,
        element_symbols,
        run_loc,
        geometry,
        ad_cont_opts,
        atom_specifier,
        extra_basis,
    ):
        self.element_symbols = element_symbols
        self.run_loc = run_loc
        self.geometry = geometry
        self.ad_cont_opts = ad_cont_opts
        self.atom_specifier = atom_specifier
        self.extra_basis = extra_basis
        self.current_path = os.path.dirname(os.path.realpath(__file__))

        # Convert k_grid key to a string from a tuple
        # Writing the options for a hole calculation doesn't use ASE, so it must be
        # converted to a string here
        if "k_grid" in ad_cont_opts:
            ad_cont_opts["k_grid"] = " ".join(map(str, ad_cont_opts["k_grid"]))

        self.new_control = f"{self.run_loc}/ground/control.in.new"
        self.elements = utils.get_all_elements()
        self.azimuthal_refs = {"s": 1, "p": 2, "d": 3, "f": 4, "g": 5}

    def calculate_highest_E_orbital(self, orb_list: list[str]) -> str:
        """
        Determine the highest energy orbital according to the Madelung rule.

        Parameters
        ----------
        orb_list : List[str]
            list of orbitals

        Returns
        -------
        str
           highest energy orbital
        """
        madelung_1 = [int(i[0]) + self.azimuthal_refs[i[1]] for i in orb_list]
        max_i_vals = [i for i, j in enumerate(madelung_1) if j == max(madelung_1)]

        # If multiple max values, take the one with the highest n
        max_n_vals = [int(orb_list[i][0]) for i in max_i_vals]
        orbs = orb_list[max_i_vals[max_n_vals.index(max(max_n_vals))]]

        return f"    valence      {orbs[0]}  {orbs[1]}   {orbs[2:]}\n"

    def get_electronic_structure(self, atom: str) -> str:
        """
        Get valence electronic structure of target atom.

        Adapted from scipython.com question P2.5.12.

        Parameters
        ----------
        atom : str
            element symbol of target atom

        Returns
        -------
        valence : str
            valence electronic structure of target atom
        """
        # Get the atomic number of the target atom
        self.atom_index = self.elements.index(str(atom)) + 1

        # Create and order a list of tuples, (n+l, n, l), corresponding to the order
        # in which the corresponding orbitals are filled using the Madelung rule.
        nl_pairs = []

        for n in range(1, 8):
            for l in range(n):
                nl_pairs.append((n + l, n, l))

        nl_pairs.sort()

        inl = 0
        n_elec = 0
        n = 1
        l = 0
        config = [["1s", 0]]
        noble_gas_config = ("", "")
        s_config = ""
        l_letter = ["s", "p", "d", "f", "g"]

        for i in range(len(self.elements[: self.atom_index])):
            n_elec += 1

            if n_elec > 2 * (2 * l + 1):  # Subshell full
                if l == 1:
                    # Save this noble gas configuration
                    noble_gas_config = (
                        ".".join(["{:2s}{:d}".format(*e) for e in config]),
                        f"[{self.elements[i - 1]}]",
                    )

                # Start a new subshell
                inl += 1
                _, n, l = nl_pairs[inl]
                config.append([f"{n}{l_letter[l]}", 1])
                n_elec = 1

            # add an electron to the current subshell
            else:
                config[-1][1] += 1

            s_config = ".".join(["{:2s}{:d}".format(*e) for e in config])
            s_config = s_config.replace(*noble_gas_config)

        # Find the orbital with the highest energy according to the Madelung rule
        output = list(s_config.split(".")[1:])
        self.valence = self.calculate_highest_E_orbital(output)

        return self.valence

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

    def add_partial_charge(
        self, content, target_atom, at_num, partial_charge
    ) -> tuple[int, int, str, list[str]]:
        """
        Add a partial charge to a basis set in a control.in file.

        Parameters
        ----------
        content : List[str]
            list of lines in the control file
        target_atom : str
            element symbol of target atom
        at_num : int
            atomic number of target atom
        partial_charge : float
            partial charge to add to the nucleus

        Returns
        -------
        nuclear_index : int
            index of the nucleus in the control file
        nucleus : str
            nucleus line in the control file
        content : List[str]
            list of lines in the control file
        """
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
                    content[nuclear_index] = (
                        f"    nucleus             {at_num + partial_charge}\n"
                    )
                elif f"    nucleus      {at_num}\n" in content[j:]:
                    nuclear_index = (
                        content[j:].index(f"    nucleus      {at_num}\n") + j
                    )
                    nucleus = content[nuclear_index]  # save for hole
                    content[nuclear_index] = (
                        f"    nucleus      {at_num + partial_charge}\n"
                    )

                # Add to valence orbital
                if "#     ion occupancy\n" in content[j:]:
                    vbs_index = content[j:].index("#     valence basis states\n") + j
                    io_index = content[j:].index("#     ion occupancy\n") + j

                    # Check which orbital to add 0.1 to
                    valence_structure = []
                    i_orbital = []

                    for count, valence_orbital in enumerate(
                        content[vbs_index + 1 : io_index]
                    ):
                        i_orbital.append(count)
                        valence_structure.append("".join(valence_orbital.split()[1:]))

                    valence = self.calculate_highest_E_orbital(valence_structure)
                    addition_state = valence_structure.index(
                        "".join(valence.split()[1:])
                    )

                    # Add the 0.1 electron
                    valence_index = vbs_index + addition_state + 1
                    content[valence_index] = valence.strip("\n") + "1\n"
                    break

        return nuclear_index, valence_index, nucleus, content


class Projector(ForceOccupation):
    """
    Create input files for projector calculations.

    ...

    Attributes
    ----------
    TODO
    """

    def __init__(self, parent_instance):
        # Inherit all the variables from an instance of the parent class
        vars(self).update(vars(parent_instance))

    def setup_init_1(self, basis_type: str, defaults: str, control: str):
        """
        Write new directories and control files for the first init calculation.

        Parameters
        ----------
        basis_type : str
            Type of the basis set to use
        defaults : str
            Default location of the basis sets
        control : str
            Path to the control file
        """
        # Default control file options
        opts = {
            "charge": 0.1,
            "sc_iter_limit": 500,
            "sc_init_iter": 75,
            "restart_write_only": "restart_file",
            "restart_save_iterations": 5,
            # "force_single_restartfile": ".true.",
        }

        # Add or change user-specified keywords to/in the control file
        opts = self.mod_keywords(self.ad_cont_opts, opts)

        # Create a new intermediate file and write basis sets to it
        shutil.copyfile(control, self.new_control)

        # Loop over each element to constrain
        for el in self.element_symbols:
            # Find species defaults location from location of binary
            basis_set = glob.glob(
                f"{defaults}/defaults_2020/{basis_type}/*{el}_default"
            )
            # Append the contents of the found basis set file to self.new_control
            if basis_set:
                with (
                    open(basis_set[0]) as basis_file,
                    open(self.new_control, "a") as new_control,
                ):
                    shutil.copyfileobj(basis_file, new_control)
            else:
                warnings.warn(
                    f"No basis set file found for element {el} in {defaults}/"
                    f"defaults_2020/{basis_type}",
                    stacklevel=2,
                )

            # Change basis set label for core hole atom
            with open(self.new_control) as read_control:
                new_basis_content = read_control.readlines()

            for j, line in enumerate(new_basis_content):
                spl = line.split()

                if len(spl) > 1 and spl[0] == "species" and el == spl[1]:
                    new_basis_content[j] = f"  species        {el}1\n"
                    break

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
                shutil.copyfile(self.new_control, i1_control)
                # Create new geometry file for init_1 calc
                shutil.copyfile(f"{self.run_loc}/ground/geometry.in", i1_geometry)

                # Change geometry file
                with open(i1_geometry) as read_geom:
                    geom_content = read_geom.readlines()

                # Change all core hole atoms to {atom}{num}
                atom_counter = 0
                for j, line in enumerate(geom_content):
                    spl = line.split()

                    if "atom" in line and el in line:
                        # if atom_counter + 1 == i:
                        if atom_counter + 1 in self.atom_specifier:
                            partial_hole_atom = f" {el}1\n"
                            geom_content[j] = " ".join(spl[0:-1]) + partial_hole_atom

                        atom_counter += 1

                with open(i1_geometry, "w") as write_geom:
                    write_geom.writelines(geom_content)

                # Change control file
                control_content = self.change_control_keywords(i1_control, opts)

                # Add additional core-hole basis functions
                if self.extra_basis:
                    control_content = self.add_additional_basis(
                        self.elements, control_content, f"{el}1"
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
                    opts["charge"],
                )

                with open(i1_control, "w") as write_control:
                    write_control.writelines(control_content)

        print("init_1 files written successfully")

    def setup_init_2(self, ks_start, ks_stop, occ, occ_type, spin, pbc):
        """Write new directories and control files for the second init calculation."""
        ks_method = ""
        if occ_type == "force_occupation_projector":
            ks_method = "serial"
        if occ_type == "deltascf_projector" and not pbc:
            ks_method = "parallel"
        if occ_type == "deltascf_projector" and pbc:
            ks_method = "serial"

        # Loop over each element to constrain
        for el in self.element_symbols:
            # Loop over each individual atom to constrain
            for i in self.atom_specifier:
                opts = {
                    "spin": "collinear",
                    "charge": 1.1,
                    "sc_iter_limit": 1,
                    occ_type: f"{i + ks_start - 1} {spin} {occ} {ks_start} {ks_stop}",
                    "KS_method": ks_method,
                    "restart": "restart_file",
                    "restart_save_iterations": 1,
                    # "force_single_restartfile": ".true.",
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
                if self.extra_basis:
                    control_content = self.add_additional_basis(
                        self.elements, control_content, f"{el}1"
                    )

                # Add partial charge to the control file
                _, _, _, control_content = self.add_partial_charge(
                    control_content, el, self.elements.index(el) + 1, opts["charge"] - 1
                )

                with open(i2_control, "w") as write_control:
                    write_control.writelines(control_content)

        print("init_2 files written successfully")

    def setup_hole(self, ks_start, ks_stop, occ, occ_type, spin, pbc):
        """Write new hole directories and control files for the hole calculation."""
        # Calculate original valence state
        val_spl = self.valence.split("\n")
        del val_spl[-1]
        val_spl.append(".\n")
        self.valence = "".join(val_spl)

        # Enforce that the old method uses ks_method serial
        ks_method = ""
        if occ_type == "force_occupation_projector":
            ks_method = "serial"

        ks_method = "serial" if occ_type == "deltascf_projector" and pbc else None

        # Loop over each element to constrain
        for el in self.element_symbols:
            # Loop over each individual atom to constrain
            for i in self.atom_specifier:
                opts = {
                    "spin": "collinear",
                    "charge": 1.0,
                    "sc_iter_limit": 500,
                    "sc_init_iter": 75,
                    occ_type: f"{i + ks_start - 1} {spin} {occ} {ks_start} {ks_stop}",
                    "KS_method": ks_method,
                    "restart": "restart_file",
                    # "force_single_restartfile": ".true.",
                    # "output": "cube spin_density",
                }

                if ks_method is not None:
                    opts["KS_method"] = ks_method

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

                with open(h_control) as read_control:
                    control_content = read_control.readlines()

                # Set nuclear and valence orbitals back to integer values
                control_content[self.n_index] = self.nucleus
                control_content[self.v_index] = self.valence

                # Change control file
                control_content = self.change_control_keywords(h_control, opts)

                # Add additional core-hole basis functions
                if self.extra_basis:
                    control_content = self.add_additional_basis(
                        self.elements, control_content, f"{el}1"
                    )

                # Write the data to the file
                with open(h_control, "w") as write_control:
                    write_control.writelines(control_content)

        print("hole files written successfully")


class Basis(ForceOccupation):
    """Create input files for basis calculations."""

    def __init__(self, parent_instance):
        # Inherit all the variables from an instance of the parent class
        vars(self).update(vars(parent_instance))

    def setup_basis(
        self, spin, n_qn, l_qn, m_qn, occ_no, ks_max, occ_type, basis_set, defaults
    ):
        """Write new directories and control files for basis calculations."""
        # Enforce that the old method uses ks_method serial
        ks_method = "serial" if occ_type == "force_occupation_basis" else None

        # Iterate over each constrained atom
        for el in self.element_symbols:
            # Loop over each individual atom to constrain
            for i in range(len(self.atom_specifier)):
                # Default control file options
                opts = {
                    "spin": "collinear",
                    "charge": 1.0,
                    "sc_iter_limit": 500,
                    "sc_init_iter": 75,
                    occ_type: f"{self.atom_specifier[i]} {spin} atomic {n_qn} {l_qn} "
                    f"{m_qn} {occ_no} {ks_max}",
                }

                if ks_method is not None:
                    opts["KS_method"] = ks_method

                # Allow users to modify and add keywords
                opts = self.mod_keywords(self.ad_cont_opts, opts)

                i += 1

                control = f"{self.run_loc}/{el}{i}/control.in"
                geometry = f"{self.run_loc}/{el}{i}/geometry.in"

                # Create new directories and .in files for each constrained atom
                os.makedirs(f"{self.run_loc}/{el}{i}/", exist_ok=True)
                shutil.copyfile(
                    f"{self.run_loc}/ground/control.in",
                    control,
                )
                shutil.copyfile(
                    f"{self.run_loc}/ground/geometry.in",
                    geometry,
                )

                # Create new geometry file for hole calc by adding label for
                # core hole basis set
                # Change geometry file
                with open(geometry) as read_geom:
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

                with open(geometry, "w") as write_geom:
                    write_geom.writelines(geom_content)

                # Change control file
                # Find species defaults location from location of binary and
                # add basis set to control file from defaults
                basis_set_file = glob.glob(
                    f"{defaults}/defaults_2020/{basis_set}/??_{el}_default"
                )
                os.system(f"cat {basis_set_file[0]} >> {control}")

                # Change basis set label for core hole atom
                with open(control) as read_control:
                    control_content = read_control.readlines()

                for j, line in enumerate(control_content):
                    spl = line.split()

                    if len(spl) > 1 and spl[0] == "species" and el == spl[1]:
                        control_content[j] = f"  species        {el}1\n"
                        break

                # Write the new label to the file
                with open(control, "w") as write_control:
                    write_control.writelines(control_content)

                # Change control file keywords
                control_content = self.change_control_keywords(control, opts)

                # Add additional core-hole basis functions
                if self.extra_basis:
                    control_content = self.add_additional_basis(
                        self.elements, control_content, f"{el}1"
                    )

                # Write the keywords and basis functions to the file
                with open(control, "w") as write_control:
                    write_control.writelines(control_content)

        print("files and directories written successfully")
