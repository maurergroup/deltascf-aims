import shutil
from pathlib import Path
from typing import Any, Final, Literal

from deltascf_aims.utils import control_utils, geometry_utils


class ForceOccupation:
    """
    Manipulate FHIaims input files to setup basis and projector calculations.

    ...

    Parameters
    ----------
    element : str
        Element symbol to constrain.
    run_loc : Path
        Path to the run directory.
    geometry : Path
        Path to the geometry file.
    add_constr_opts : dict[str, str]
        Additional control options.
    atom_specifier : list[int]
        List of atom indentifiers of atoms by geometry.in index.

    Attributes
    ----------
    element : str
        Element symbol to constrain
    run_loc : Path
        Path to the run directory.
    geometry : Path
        Path to the geometry file.
    add_constr_opts : dict[str, Any]
        Additional control options.
    atom_specifier : list[int]
        List of atom identifiers of atoms by geometry.in index.
    """

    AZIMUTHAL_REFS: Final[dict[str, int]] = {"s": 1, "p": 2, "d": 3, "f": 4, "g": 5}

    def __init__(
        self,
        constr_atom: str,
        run_loc: Path,
        geometry: Path,
        add_constr_opts: dict[str, str],
        atom_specifier: list[int],
    ):
        self._constr_atom = constr_atom
        self._run_loc = run_loc
        self._geometry = geometry
        self._add_constr_opts = add_constr_opts
        self._atom_specifier = atom_specifier

        self.current_path = Path(__file__).parent.resolve()

        # Convert k_grid key to a string from a tuple
        # Writing the options for a hole calculation doesn't use ASE, so it must be
        # converted to a string here
        if "k_grid" in add_constr_opts:
            add_constr_opts["k_grid"] = " ".join(map(str, add_constr_opts["k_grid"]))

        self.new_control = run_loc / "ground/control.in.new"
        self.elements = geometry_utils.get_all_elements()

    @property
    def constr_atom(self) -> str:
        return self._constr_atom

    @property
    def run_loc(self) -> Path:
        return self._run_loc

    @property
    def geometry(self) -> Path:
        return self._geometry

    @property
    def add_constr_opts(self) -> dict[str, str]:
        return self._add_constr_opts

    @property
    def atom_specifier(self) -> list[int]:
        return self._atom_specifier

    @atom_specifier.setter
    def atom_specifier(self, value: list[int]) -> None:
        self._atom_specifier = value

    def calculate_highest_E_orbital(self, orb_list: list[str]) -> str:  # noqa: N802
        """
        Determine the highest energy orbital according to the Madelung rule.

        Parameters
        ----------
        orb_list : list[str]
            list of orbitals

        Returns
        -------
        str
           highest energy orbital
        """
        madelung_1 = [int(i[0]) + self.AZIMUTHAL_REFS[i[1]] for i in orb_list]
        max_i_vals = [i for i, j in enumerate(madelung_1) if j == max(madelung_1)]

        # If multiple max values, take the one with the highest n
        max_n_vals = [int(orb_list[i][0]) for i in max_i_vals]
        orbs = orb_list[max_i_vals[max_n_vals.index(max(max_n_vals))]]

        return f"    valence      {orbs[0]}  {orbs[1]}   {orbs[2:]}\n"

    def get_electronic_structure(self, atom: str) -> str:
        """
        Get the valence electronic structure of the target atom.

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
        nl_pairs = sorted((n + ell, n, ell) for n in range(1, 8) for ell in range(n))

        inl = 0
        n_elec = 0
        n = 1
        ell = 0
        config = [["1s", 0]]
        noble_gas_config = ("", "")
        s_config = ""
        l_letter = ["s", "p", "d", "f", "g"]

        for i in range(len(self.elements[: self.atom_index])):
            n_elec += 1

            if n_elec > 2 * (2 * ell + 1):  # Subshell full
                if ell == 1:
                    # Save this noble gas configuration
                    noble_gas_config = (
                        ".".join(["{:2s}{:d}".format(*e) for e in config]),
                        f"[{self.elements[i - 1]}]",
                    )

                # Start a new subshell
                inl += 1
                _, n, ell = nl_pairs[inl]
                config.append([f"{n}{l_letter[ell]}", 1])
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

    def _find_basis_set_start(self, content: list[str], constr_species_idx: int) -> int:
        """
        Find the start index of a basis set.

        Parameters
        ----------
        content : list[str]
            List of lines in the control file
        constr_species_idx : int
            Index of the species line in the control file

        Returns
        -------
        int
            Index of the start of the basis set for the constrained atom

        Raises
        ------
        ValueError
            If the start of the basis set cannot be found.
        """
        n_hash_lines = 0
        species_start = None

        for i, r_line in enumerate(reversed(content[:constr_species_idx])):
            if "#" * 80 in r_line:
                n_hash_lines += 1

            if n_hash_lines == 2:
                species_start = constr_species_idx - i
                break

        if species_start is None:
            msg = (
                f"Could not find the start of the basis set for {self.constr_atom} "
                "in the control file."
            )
            raise ValueError(msg)

        return species_start

    def add_second_ch_basis(self, content: list[str]) -> list[str]:
        """
        Insert a second basis set for the constrained atom with a partial charge.

        Parameters
        ----------
        TODO
        """
        # Copy the basis set for the constrained atom already in the control file
        constr_species_idx = None
        next_species_idx = None

        for i, line in enumerate(content):
            spl = line.split()
            if len(spl) > 1 and spl[0] == "species" and spl[1] == self.constr_atom:
                constr_species_idx = i
                continue

            if constr_species_idx is not None and len(spl) > 1 and spl[0] == "species":
                # Get the start index of the next basis set so we know where the constr
                # basis set ends
                next_species_idx = self._find_basis_set_start(content, i)
                break

        if constr_species_idx is None:
            msg = f"Constrained atom {self.constr_atom} not found in control file."
            raise ValueError(msg)

        if next_species_idx is None:
            # If no other species are found, set the end of the constrained species
            # basis set to the end of the file
            next_species_idx = len(content)

        # Find the start idx of the constr basis set
        species_start = self._find_basis_set_start(content, constr_species_idx)

        # Append 1 to the species name
        constr_basis = content[species_start - 1 : next_species_idx]
        for i, line in enumerate(constr_basis):
            spl = line.split()
            if len(spl) > 1 and spl[0] == "species" and spl[1] == self.constr_atom:
                # Add a 1 to the end of the species name
                constr_basis[i] = line.replace(
                    f"{self.constr_atom}\n", f"{self.constr_atom}1\n"
                )
                break

        # Splice the constr basis set underneath the original
        content[next_species_idx:next_species_idx] = constr_basis
        return content

    def modify_basis_set_charge(
        self, content: list[str], target_atom: str, at_num: int, partial_charge: float
    ) -> tuple[int, int, str, list[str]]:
        """
        Add a partial charge to a basis set of the control.in content.

        Parameters
        ----------
        content : list[str]
            list of lines in the control file
        target_atom : str
            element symbol of target atom
        at_num : int
            atomic number of target atom
        partial_charge : float
            partial charge to add to the nucleus

        Returns
        -------
        int
            index of the nucleus in the control file
        int
            index of the valence orbital in the control file
        str
            nucleus line in the control file
        list[str]
            list of lines in the control file
        """
        # Ensure returned variables are bound
        nuclear_idx = 0
        valence_idx = 0
        nucleus = ""

        for j, line in enumerate(content):
            spl = line.split()

            if len(spl) > 1 and target_atom == spl[1]:
                # Add to nucleus
                if f"    nucleus             {at_num}\n" in content[j:]:
                    nuclear_idx = (
                        content[j:].index(f"    nucleus             {at_num}\n") + j
                    )
                    nucleus = content[nuclear_idx]  # save for hole
                    content[nuclear_idx] = (
                        f"    nucleus             {at_num + partial_charge}\n"
                    )
                elif f"    nucleus      {at_num}\n" in content[j:]:
                    nuclear_idx = content[j:].index(f"    nucleus      {at_num}\n") + j
                    nucleus = content[nuclear_idx]  # save for hole
                    content[nuclear_idx] = (
                        f"    nucleus      {at_num + partial_charge}\n"
                    )

                # Add to valence orbital
                if "#     ion occupancy\n" in content[j:]:
                    vbs_idx = content[j:].index("#     valence basis states\n") + j
                    io_idx = content[j:].index("#     ion occupancy\n") + j

                    # Check which orbital to add 0.1 to
                    valence_structure = []
                    i_orbital = []

                    for count, valence_orbital in enumerate(
                        content[vbs_idx + 1 : io_idx]
                    ):
                        i_orbital.append(count)
                        valence_structure.append("".join(valence_orbital.split()[1:]))

                    valence = self.calculate_highest_E_orbital(valence_structure)
                    addition_state = valence_structure.index(
                        "".join(valence.split()[1:])
                    )

                    # Add the 0.1 electron
                    valence_idx = vbs_idx + addition_state + 1
                    content[valence_idx] = valence.strip("\n") + "1\n"
                    break

        return nuclear_idx, valence_idx, nucleus, content


class Projector(ForceOccupation):
    """
    Create input files for projector calculations.

    ...

    Parameters
    ----------
    constr_atom : str
        Element symbol to constrain.
    run_loc : Path
        Path to the run directory.
    geometry : Path
        Path to the geometry file.
    add_constr_opts : dict[str, Any]
        Additional control options.
    atom_specifier : list[int]
        List of atom identifiers of atoms by geometry.in index.
    extra_basis : bool
        Whether to add extra basis functions to the control file.

    Attributes
    ----------
    constr_atom : str
        Element symbol to constrain.
    run_loc : Path
        Path to the run directory.
    geometry : Path
        Path to the geometry file.
    add_constr_opts : dict[str, Any]
        Additional control options.
    atom_specifier : list[int]
        List of atom identifiers of atoms by geometry.in index.
    extra_basis : bool
        Whether to add extra basis functions to the control file.
    """

    def __init__(self, *args: Any):
        super().__init__(*args)

    def setup_init_1(
        self,
        ks_range: tuple[int, int],
        control: Path,
        pbc: bool,
    ) -> None:
        """
        Write new directories and control files for the first init calculation.

        Parameters
        ----------
        ks_range : tuple[int, int]
            KS states to calculate MOM over.
        control : Path
            Path to the control file.
        pbc : bool
            If the calculation is periodic.
        """
        # Default control file options
        opts = {
            "charge": 0.1,
            "sc_iter_limit": 500,
            "sc_init_iter": 75,
            "restart_write_only": "restart_file",
            "restart_save_iterations": 5,
        }

        # Enforce that the ks_method serial to write appropriate restart files for dscf
        # calculations if the calculation is periodic
        if pbc:
            opts["KS_method"] = "serial"

        # Add or change user-specified keywords to/in the control file
        opts = control_utils.mod_keywords(self.add_constr_opts, opts)

        # Create a new intermediate file and write basis sets to it
        shutil.copyfile(control, self.new_control)

        # Check how many KS states to assign per atom
        ks_states = [*list(range(*ks_range)), ks_range[-1]]
        ks_states_per_atom = len(ks_states) / len(self.atom_specifier)
        print(f"Specified Kohn-Sham Eigenstates:\n{' '.join(map(str, ks_states))}\n")
        print(
            f"Constrained Kohn-Sham Eigenstates per atom:\n{int(ks_states_per_atom)}\n"
        )

        expanded_atom_specifier = []
        for atom in self.atom_specifier:
            expanded_atom_specifier.extend([atom] * int(ks_states_per_atom))

        self.atom_specifier = expanded_atom_specifier

        for atom, ks_state in zip(self.atom_specifier, ks_states, strict=False):
            i1_control = (
                self.run_loc / f"{self.constr_atom}{ks_state}/init_1/control.in"
            )
            i1_geometry = (
                self.run_loc / f"{self.constr_atom}{ks_state}/init_1/geometry.in"
            )

            # Create new directory and control file for init_1 calc
            (self.run_loc / f"{self.constr_atom}{ks_state}" / "init_1").mkdir(
                parents=True, exist_ok=True
            )
            shutil.copyfile(self.new_control, i1_control)

            # Create new geometry file for init_1 calc
            shutil.copyfile(self.run_loc / "ground/geometry.in", i1_geometry)

            # Find the correct atom idx to constrain
            with i1_geometry.open() as geom:
                geom_content = geom.readlines()

            found_atom = False
            atom_idx = 0

            for j, line in enumerate(geom_content):
                spl = line.split()
                if len(spl) > 1 and spl[0] == "atom":
                    atom_idx += 1
                    if spl[-1] == self.constr_atom and atom_idx == atom:
                        # Add 1 to the end of the atom line
                        geom_content[j] = f"{line[:-1]}1\n"
                        found_atom = True
                        break

            if not found_atom:
                msg = (
                    f"Atom {self.constr_atom} with index {atom} not found in "
                    "geometry.in file."
                )
                raise ValueError(msg)

            # Change control file
            control_content = control_utils.change_control_keywords(i1_control, opts)
            control_content = self.add_second_ch_basis(control_content)

            # Add partial charge to the control file
            (
                self.n_index,
                self.v_index,
                self.nucleus,
                control_content,
            ) = self.modify_basis_set_charge(
                control_content,
                self.constr_atom + "1",
                self.elements.index(self.constr_atom) + 1,
                float(opts["charge"]),
            )

            # Write the modified input files
            with i1_control.open("w") as write_control:
                write_control.writelines(control_content)

            with i1_geometry.open("w") as write_geometry:
                write_geometry.writelines(geom_content)

        print("Wrote init_1 files")

    def setup_init_2(
        self,
        ks_range: tuple[int, int],
        occ: float,
        occ_type: Literal["deltascf_projector", "force_occupation_projector"],
        spin: Literal[1, 2],
        pbc: bool,
    ) -> None:
        """
        Write new directories and control files for the second init calculation.

        Parameters
        ----------
        ks_range : tuple[int, int]
            KS states to calculate MOM over.
        occ : float
            Occupation value to constrain specified state to.
        occ_type : Literal["deltascf_projector", "force_occupation_projector"]
            FHI-aims projector keyword.
        spin : Literal[1, 2]
            Spin channel of the KS state to constrain.
        pbc : bool
            If the calculation is periodic.
        """
        # Loop over each KS state to constrain
        for i in [*list(range(*ks_range)), ks_range[-1]]:
            opts = {
                "spin": "collinear",
                "charge": 1.1,
                "sc_iter_limit": 1,
                occ_type: f"{i} {spin} {occ} {ks_range[0]} {ks_range[1]}",
                "restart": "restart_file",
                "restart_save_iterations": 1,
            }

            # Enforce that the use of ks_method serial
            if occ_type == "force_occupation_projector" or (
                occ_type == "deltascf_projector" and pbc
            ):
                opts["KS_method"] = "serial"

            # Add or change user-specified keywords to the control file
            opts = control_utils.mod_keywords(self.add_constr_opts, opts)

            i2_control = f"{self.run_loc}/{self.constr_atom}{i}/init_2/control.in"

            # Create new directory for init_2 calc
            init_2_dir = self.run_loc / f"{self.constr_atom}{i}/init_2"
            init_2_dir.mkdir(parents=True, exist_ok=True)
            i2_control = init_2_dir / "control.in"
            shutil.copyfile(self.new_control, i2_control)
            shutil.copyfile(
                self.run_loc / f"{self.constr_atom}{i}/init_1/geometry.in",
                init_2_dir / "geometry.in",
            )

            # Change control file
            control_content = control_utils.change_control_keywords(i2_control, opts)
            control_content = self.add_second_ch_basis(control_content)

            # Add full + partial charge to the control file
            _, _, _, control_content = self.modify_basis_set_charge(
                control_content,
                self.constr_atom + "1",
                self.elements.index(self.constr_atom) + 1,
                float(opts["charge"]) - 1,
            )

            with i2_control.open("w") as write_control:
                write_control.writelines(control_content)

        print("Wrote init_2 files")

    def setup_hole(
        self,
        ks_range: tuple[int, int],
        occ: float,
        occ_type: Literal["deltascf_projector", "force_occupation_projector"],
        spin: Literal[1, 2],
        pbc: bool,
    ) -> None:
        """
        Write new hole directories and control files for the hole calculation.

        Parameters
        ----------
        ks_range : tuple[int, int]
            KS states to calculate MOM over.
        occ : float
            Occupation value to constrain specified state to.
        occ_type : Literal["deltascf_projector", "force_occupation_projector"]
            FHI-aims projector keyword.
        spin : Literal[1, 2]
            Spin channel of the KS state to constrain.
        pbc : bool
            If the calculation is periodic.
        """
        # Calculate original valence state
        val_spl = self.valence.split("\n")
        del val_spl[-1]
        val_spl.append(".\n")
        self.valence = "".join(val_spl)

        # Loop over each KS state to constrain
        for i in [*list(range(*ks_range)), ks_range[-1]]:
            opts = {
                "spin": "collinear",
                "charge": 1.0,
                "sc_iter_limit": 500,
                "sc_init_iter": 75,
                occ_type: f"{i} {spin} {occ} {ks_range[0]} {ks_range[1]}",
                "restart": "restart_file",
                "output": "mulliken",
            }

            # Enforce that the use of ks_method serial
            if occ_type == "force_occupation_projector" or (
                occ_type == "deltascf_projector" and pbc
            ):
                opts["KS_method"] = "serial"

            # Add or change user-specified keywords to the control file
            opts = control_utils.mod_keywords(self.add_constr_opts, opts)

            # Location of the hole control file
            h_control = self.run_loc / f"{self.constr_atom}{i}/hole/control.in"

            # Create new directory for hole calc
            (self.run_loc / f"{self.constr_atom}{i}" / "hole").mkdir(
                parents=True, exist_ok=True
            )
            shutil.copyfile(
                self.run_loc / f"{self.constr_atom}{i}/init_1/geometry.in",
                self.run_loc / f"{self.constr_atom}{i}/hole/geometry.in",
            )
            shutil.copyfile(self.new_control, h_control)

            with h_control.open() as read_control:
                control_content = read_control.readlines()

            # Change control file
            control_content = control_utils.change_control_keywords(h_control, opts)
            control_content = self.add_second_ch_basis(control_content)

            # Set nuclear and valence orbitals back to integer values
            control_content[self.n_index] = self.nucleus
            control_content[self.v_index] = self.valence

            # Write the data to the file
            with h_control.open("w") as write_control:
                write_control.writelines(control_content)

        print("Wrote hole files")


class Basis(ForceOccupation):
    """
    Create input files for basis calculations.

    ...

    Parameters
    ----------
    constr_atom : str
        list of element symbols to constrain
    run_loc : Path
        Path to the run directory.
    geometry : Path
        Path to the geometry file.
    add_constr_opts : dict[str, Any]
        Additional control options.
    atom_specifier : list[int]
        List of atom identifiers of atoms by geometry.in index.

    Attributes
    ----------
    constr_atom : str
        list of element symbols to constrain
    run_loc : Path
        Path to the run directory.
    geometry : Path
        Path to the geometry file.
    add_constr_opts : dict[str, Any]
        Additional control options.
    atom_specifier : list[int]
        List of atom identifiers of atoms by geometry.in index.
    """

    def __init__(self, *args: Any):
        super().__init__(*args)

    def setup_basis(
        self,
        spin: Literal[1, 2],
        n_qn: int,
        l_qn: int,
        m_qn: int,
        occ_no: float,
        ks_max: int,
        occ_type: Literal["deltascf_basis", "force_occupation_basis"],
    ) -> None:
        """
        Write new directories and control files for basis calculations.

        Parameters
        ----------
        spin : Literal[1, 2]
            Spin channel of the KS state to constrain.
        n_qn : int
            Principal quantum number of the orbital to constrain.
        l_qn : int
            Azimuthal quantum number of the orbital to constrain.
        m_qn : int
            Magnetic quantum number of the orbital to constrain.
        occ_no : float
            Occupation number of the orbital to constrain.
        ks_max : int
            Maximum KS state to include in the MOM.
        occ_type : Literal["deltascf_basis", "force_occupation_basis"]
            FHI-aims basis keyword.
        """
        # Enforce that the old method uses ks_method serial
        ks_method = "serial" if occ_type == "force_occupation_basis" else None

        # Loop over each individual atom to constrain
        for i, atom_idx in enumerate(self.atom_specifier):
            # Default control file options
            opts = {
                "spin": "collinear",
                "charge": 1.0,
                "sc_iter_limit": 500,
                "sc_init_iter": 75,
                occ_type: f"{atom_idx} {spin} atomic {n_qn} {l_qn} "
                f"{m_qn} {occ_no} {ks_max}",
            }

            if ks_method is not None:
                opts["KS_method"] = ks_method

            # Allow users to modify and add keywords
            opts = control_utils.mod_keywords(self.add_constr_opts, opts)

            i += 1  # noqa: PLW2901

            control = self.run_loc / f"{self.constr_atom}{atom_idx}/control.in"
            geometry = self.run_loc / f"{self.constr_atom}{atom_idx}/geometry.in"

            # Create new directories and .in files for each constrained atom
            (self.run_loc / f"{self.constr_atom}{atom_idx}").mkdir(
                parents=True, exist_ok=True
            )
            shutil.copyfile(
                self.run_loc / "ground/control.in",
                control,
            )
            shutil.copyfile(
                self.run_loc / "ground/geometry.in",
                geometry,
            )

            # Change control file keywords
            control_content = control_utils.change_control_keywords(control, opts)

            # Write the keywords to the file
            with control.open("w") as write_control:
                write_control.writelines(control_content)

        print("Wrote hole files")
