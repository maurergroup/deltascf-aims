"""Utilities which are used in aims_dscf"""

import glob
import os
import sys

import yaml
from ase.build import molecule
from ase.calculators.aims import Aims
from ase.data.pubchem import pubchem_atoms_search
from ase.io import write
from click import MissingParameter
from delta_scf.force_occupation import ForceOccupation as fo


class MainUtils:
    """Various utilities used in aims_dscf"""

    @staticmethod
    def build_geometry(geometry):
        """Check different databases to create a geometry.in"""

        try:
            atoms = molecule(geometry)
            print("molecule found in ASE database")
            return atoms
        except KeyError:
            print("molecule not found in ASE database, searching PubChem...")

        try:
            atoms = pubchem_atoms_search(name=geometry)
            print("molecule found as a PubChem name")
            return atoms
        except ValueError:
            print(f"{geometry} not found in PubChem name")

        try:
            atoms = pubchem_atoms_search(cid=geometry)
            print("molecule found in PubChem CID")
            return atoms
        except ValueError:
            print(f"{geometry} not found in PubChem CID")

        try:
            atoms = pubchem_atoms_search(smiles=geometry)
            print("molecule found in PubChem SMILES")
            return atoms
        except ValueError:
            print(f"{geometry} not found in PubChem smiles")
            print(f"{geometry} not found in PubChem or ASE database")
            print("aborting...")
            sys.exit(1)

    @staticmethod
    def check_args(*args):
        """Check if required arguments are specified"""

        def_args = locals()

        for arg in def_args["args"]:
            if arg[1] is None:
                if arg[0] == "spec_mol":
                    # Convert to list and back to assign to tuple
                    arg = list(arg)
                    arg[0] = "molecule"
                    arg = tuple(arg)

                raise MissingParameter(param_hint=f"'--{arg[0]}'", param_type="option")

    @staticmethod
    def check_geom(geom_file):
        """Check if there are any constrain_relaxation keywords in geometry.in"""

        lattice_vecs = False

        for line in geom_file:
            if "lattice_vector" in line:
                lattice_vecs = True

            if "constrain_relaxation" in line:
                print("'constrain_relaxation' keyword found in geometry.in")
                print("Ensure that no atoms are fixed in the geometry.in file")
                print(
                    "The geometry of the structure should have already been relaxed before any SP calculations"
                )
                print("Aborting...")
                sys.exit(1)

        return lattice_vecs

    @staticmethod
    def check_control(control_file):
        """Check if there is a k_grid in the control.in"""

        k_grid = False

        for line in control_file:
            if "k_grid" in line:
                k_grid = True

        return k_grid

    @staticmethod
    def convert_opts_to_dict(opts, pbc):
        """Convert the control options from a tuple to a dictionary"""

        opts_dict = {}

        for opt in opts:
            spl = opt.split(sep="=")

            opts_dict[spl[0]] = spl[1]

        # Also add k_grid if given
        if pbc is not None:
            opts_dict.update({"k_grid": pbc})

        return opts_dict

    @staticmethod
    def create_calc(procs, binary, species, int_grid):
        """Create an ASE calculator object"""

        # Choose some sane defaults
        aims_calc = Aims(
            xc="pbe",
            spin="collinear",
            default_initial_moment=0,
            aims_command=f"mpirun -n {procs} {binary}",
            species_dir=f"{species}/defaults_2020/{int_grid}/",
        )

        return aims_calc

    @staticmethod
    def print_ks_states(run_loc):
        """Print the KS states for the different spin states"""

        # Parse the output file
        with open(f"{run_loc}/ground/aims.out", "r") as aims:
            lines = aims.readlines()

        su_eigs_start_line = None
        sd_eigs_start_line = None

        for num, content in enumerate(lines):
            if "Spin-up eigenvalues" in content:
                su_eigs_start_line = num
                if "K-point:" in lines[num + 1]:
                    su_eigs_start_line += 1

            if "Spin-down eigenvalues" in content:
                sd_eigs_start_line = num
                if "K-point:" in lines[num + 1]:
                    sd_eigs_start_line += 1

        # Check that KS states were found
        if su_eigs_start_line is None:
            print("No spin-up KS states found")
            print("Did you run a spin polarised calculation?")
            sys.exit(1)

        if sd_eigs_start_line is None:
            print("No spin-down KS states found")
            print("Did you run a spin polarised calculation?")
            sys.exit(1)

        su_eigs = []
        sd_eigs = []

        # Save the KS states into lists
        for num, content in enumerate(lines[su_eigs_start_line + 2 :]):
            spl = content.split()

            if len(spl) != 0:
                su_eigs.append(content)
            else:
                break

        for num, content in enumerate(lines[sd_eigs_start_line + 2 :]):
            spl = content.split()

            if len(spl) != 0:
                sd_eigs.append(content)
            else:
                break

        # Print the KS states
        print("Spin-up KS eigenvalues:\n")
        print(*su_eigs, sep="")

        print("Spin-down KS eigenvalues:\n")
        print(*sd_eigs, sep="")

    @staticmethod
    def write_control(run_loc, control_opts, atoms, int_grid, defaults):
        """Write a control.in file"""

        # Firstly create the control file
        os.system(f"touch {run_loc}/ground/control.in")

        # Convert any keys given as tuples to strings
        for i in control_opts.items():
            if type(i[1]) == tuple:
                control_opts[i[0]] = " ".join(str(j) for j in i[1])

        # Use the static method from ForceOccupation
        lines = fo.change_control_keywords(f"{run_loc}/ground/control.in", control_opts)

        with open(f"{run_loc}/ground/control.in", "w") as control:
            control.writelines(lines)

        # Then add the basis set
        elements = list(set(atoms.get_chemical_symbols()))

        for el in elements:
            basis_set = glob.glob(f"{defaults}/ch_basis_sets/{int_grid}/*{el}_default")[
                0
            ]
            os.system(f"cat {basis_set} >> {run_loc}/ground/control.in")

    @staticmethod
    def ground_calc(
        run_loc,
        geom_inp,
        control_inp,
        atoms,
        basis_set,
        species,
        calc,
        ase,
        control_opts,
        constr_atom,
        nprocs,
        binary,
        hpc,
        print_output,
    ):
        """Run a ground state calculation"""

        # Ensure that aims always runs with the following environment variables:
        os.system("export OMP_NUM_THREADS=1")
        os.system("export MKL_NUM_THREADS=1")
        os.system("export MKL_DYNAMIC=FALSE")
        os.system("ulimit -s unlimited")

        # Create the ground directory if it doesn't already exist
        os.system(f"mkdir -p {run_loc}/ground")

        # Write the geometry file if the system is specified through CLI
        if geom_inp is None and control_inp is not None:
            write(f"{run_loc}/geometry.in", atoms, format="aims")

        # Copy the geometry.in and control.in files to the ground directory
        if control_inp is not None:
            os.system(f"cp {control_inp} {run_loc}/ground")

        if geom_inp is not None:
            os.system(f"cp {geom_inp} {run_loc}/ground")

        if os.path.isfile(f"{run_loc}/ground/aims.out") is False:
            # Run the ground state calculation

            # Add additional basis functions
            basis_file = glob.glob(
                f"{species}/defaults_2020/{basis_set}/*{constr_atom}_default"
            )[0]
            current_path = os.path.dirname(os.path.realpath(__file__))

            with open(basis_file, "r") as basis_functions:
                control_content = basis_functions.readlines()

            with open(f"{current_path}/../delta_scf/elements.yml", "r") as elements:
                elements = yaml.load(elements, Loader=yaml.SafeLoader)

            new_content = fo.add_additional_basis(
                current_path, elements, control_content, constr_atom
            )

            # Create a new directory for modified basis sets
            new_dir = f"{species}/ch_basis_sets/{basis_set}/"

            if os.path.exists(new_dir):
                os.system(f"rm {new_dir}/*")
            else:
                os.system(f"mkdir -p {species}/ch_basis_sets/{basis_set}/")

            os.system(f"cp {basis_file} {species}/ch_basis_sets/{basis_set}/")
            new_basis_file = glob.glob(
                f"{species}/ch_basis_sets/{basis_set}/*{constr_atom}_default"
            )[0]

            if new_content is not None:
                with open(new_basis_file, "w") as basis_functions:
                    basis_functions.writelines(new_content)

            # Copy atoms from the original basis set directory to the new one
            chem_symbols = list(set(atoms.get_chemical_symbols()))

            for atom in chem_symbols:
                if atom != constr_atom:
                    os.system(
                        f"cp {species}/defaults_2020/{basis_set}/*{atom}_default "
                        f"{species}/ch_basis_sets/{basis_set}/"
                    )

            if ase:
                # Change the defaults if any are specified by the user
                # Update with all control options from the calculator
                calc.set(**control_opts)
                control_opts = calc.parameters

                if not hpc:
                    print("running calculation...")
                    atoms.get_potential_energy()
                    # Move files to ground directory
                    os.system(f"cp geometry.in control.in {run_loc}/ground/")
                    os.system(f"mv aims.out parameters.ase {run_loc}/ground/")

                    # print("ground calculation completed successfully\n")

                else:
                    # Prevent species dir from being written
                    control_opts.pop("species_dir")

                    print("writing geometry.in file...")
                    write(f"{run_loc}/ground/geometry.in", images=atoms, format="aims")

                    print("writing control.in file...")
                    MainUtils.write_control(
                        run_loc, control_opts, atoms, basis_set, species
                    )

            elif not hpc:  # Don't use ASE
                print("running calculation...")

                if print_output:  # Show live output of calculation
                    os.system(
                        f"cd {run_loc}/ground && mpirun -n {nprocs} {binary} | tee aims.out"
                    )

                else:
                    os.system(
                        f"cd {run_loc}/ground && mpirun -n {nprocs} {binary} > aims.out"
                    )

                # print("ground calculation completed successfully\n")

            # Print the KS states from aims.out so it is easier to specify the
            # KS states for the hole calculation
            if not hpc:
                MainUtils.print_ks_states(run_loc)

        else:
            print("aims.out file found in ground calculation directory")
            print("skipping calculation...")

    @staticmethod
    def get_element_symbols(geom, spec_at_constr):
        """Find the element symbols from specified atom indices in a geometry file."""

        with open(geom, "r") as geom:
            lines = geom.readlines()

        atom_lines = []

        # Copy only the lines which specify atom coors into a new list
        for line in lines:
            spl = line.split()
            if len(line) > 0 and "atom" == spl[0]:
                atom_lines.append(line)

        element_symbols = []

        # Get the element symbols from the atom coors
        # Uniquely add each element symbol
        for atom in spec_at_constr:
            element = atom_lines[atom].split()[-1]

            if element not in element_symbols:
                element_symbols.append(element)

        return element_symbols
