#!/usr/bin/env python3

import glob
import os
import sys
from pathlib import Path

import click
from ase import Atoms
from ase.io import read
from utils.main_utils import MainUtils as mu

from delta_scf.calc_dscf import CalcDeltaSCF as cds
from delta_scf.force_occupation import Basis, ForceOccupation, Projector
from delta_scf.plot import Plot
from delta_scf.schmid_pseudo_voigt import broaden


def main(
    ctx,
    hpc,
    geometry_input,
    control_input,
    binary,
    run_location,
    spec_mol,
    constr_atom,
    spec_at_constr,
    occupation,
    n_atoms,
    basis_set,
    graph,
    print_output,
    nprocs,
    debug,
):
    """
    Point of controlling the program flow. Options used for all methods are
    also defined here.
    """

    # Pass global options to subcommands
    ctx.ensure_object(dict)

    # A geometry file must be given if specific atom indices are to be constrained
    if len(spec_at_constr) > 0:
        if not geometry_input:
            raise click.MissingParameter(
                param_hint="-e/--geometry_input", param_type="option"
            )

    # Use ASE unless both a custom geometry.in and control.in are specified
    # Also don't use ASE if a control.in is specified
    ase = True
    found_lattice_vecs = False
    if geometry_input is not None:
        found_lattice_vecs = mu.check_geom(geometry_input)
        ctx.obj["GEOM_INP"] = geometry_input.name
    else:
        ctx.obj["GEOM_INP"] = None
        ctx.obj["LATTICE_VECS"] = None

    found_k_grid = False
    if control_input is not None:
        ase = False
        found_k_grid = mu.check_control(control_input)
        ctx.obj["CONTROL_INP"] = control_input.name
    else:
        ctx.obj["CONTROL_INP"] = None

    if found_lattice_vecs or found_k_grid:
        ctx.obj["LATTICE_VECS"] = True

    # Find the structure if not given
    # Build the structure if given
    atoms = Atoms()

    if spec_mol is None and geometry_input is None:
        if "--help" not in sys.argv:
            try:
                atoms = read(f"./{run_location}/ground/geometry.in")
                print(
                    "molecule argument not provided, defaulting to using existing geometry.in"
                    " file"
                )
            except FileNotFoundError:
                raise click.MissingParameter(
                    param_hint="-m/--molecule or -e/--geometry_input",
                    param_type="option",
                )

    elif "--help" not in sys.argv and ase:
        if spec_mol is not None:
            atoms = mu.build_geometry(spec_mol)
        if geometry_input is not None:
            atoms = read(geometry_input.name)

    # Get the constrained atom element
    # TODO: support for multiple constrained atoms
    if constr_atom is None:
        for atom in atoms:
            if atom.index in spec_at_constr:
                constr_atom = atom.symbol
                break

    # Check if a binary has been specified
    if "--help" not in sys.argv:
        current_path = os.path.dirname(os.path.realpath(__file__))
        with open(f"{current_path}/aims_bin_loc.txt", "r") as f:
            try:
                bin_path = f.readlines()[0][:-1]
            except IndexError:
                bin_path = ""

        # Ensure the user has entered the path to the binary
        # If not open the user's $EDITOR to allow them to enter the path
        if not Path(bin_path).is_file() or binary or bin_path == "":
            marker = (
                "\n# Enter the path to the FHI-aims binary above this line\n"
                "# Ensure that the binary is located in the build directory of FHIaims"
            )
            bin_line = click.edit(marker)
            if bin_line is not None:
                with open(f"{current_path}/aims_bin_loc.txt", "w") as f:
                    f.write(bin_line)

                with open(f"{current_path}/aims_bin_loc.txt", "r") as f:
                    binary = f.readlines()[0]

            else:
                raise FileNotFoundError(
                    "path to the FHI-aims binary could not be found"
                )

        elif Path(bin_path).exists():
            print(f"specified binary path: {bin_path}")
            binary = bin_path

        species = f"{Path(binary).parent.parent}/species_defaults/"

        # Check if the species_defaults directory exists in the correct location
        if not Path(species).exists():
            print(
                "\nError: ensure the FHI-aims binary is in the 'build' directory of the FHI-aims"
                " source code directory, and that the 'species_defaults' directory exists"
            )
            raise NotADirectoryError(
                f"species_defaults directory not found in {Path(binary).parent.parent}"
            )

        # Create the ASE calculator
        if ase:
            aims_calc = mu.create_calc(nprocs, binary, species, basis_set)
            atoms.calc = aims_calc
            ctx.obj["CALC"] = aims_calc

        # User specified context objects
        ctx.obj["ATOMS"] = atoms
        ctx.obj["SPEC_MOL"] = spec_mol
        ctx.obj["BINARY"] = binary
        ctx.obj["RUN_LOC"] = run_location
        ctx.obj["CONSTR_ATOM"] = constr_atom
        ctx.obj["SPEC_AT_CONSTR"] = spec_at_constr
        ctx.obj["OCC"] = occupation
        ctx.obj["N_ATOMS"] = n_atoms  # TODO
        ctx.obj["BASIS_SET"] = basis_set
        ctx.obj["GRAPH"] = graph
        ctx.obj["PRINT"] = print_output
        ctx.obj["NPROCS"] = nprocs
        ctx.obj["DEBUG"] = debug
        ctx.obj["HPC"] = hpc

        # Context objects created in main()
        ctx.obj["SPECIES"] = species
        ctx.obj["ASE"] = ase


def process(
    ctx, intensity=1, asym=False, a=0.2, b=0.0, gl_ratio=0.5, omega=0.35, gmp=0.003
):
    """Calculate DSCF values and plot the simulated XPS spectra."""

    # Calculate the delta scf energies
    grenrgys = cds.read_ground(ctx.obj["RUN_LOC"])
    element, excienrgys = cds.read_atoms(
        ctx.obj["RUN_LOC"], ctx.obj["CONSTR_ATOM"], cds.contains_number
    )
    xps = cds.calc_delta_scf(element, grenrgys, excienrgys)

    if ctx.obj["RUN_LOC"] != "./":
        os.system(f"mv {element}_xps_peaks.txt {ctx.obj['RUN_LOC']}")

    if ctx.obj["GRAPH"]:
        # Apply the peak broadening
        peaks, domain = broaden(0, 1000, intensity, gl_ratio, xps, omega, asym, a, b)

        # Write out the spectrum to a text file
        # Include bin width of 0.01 eV
        data = []
        bin_val = 0.00
        for i, peak in enumerate(peaks):
            data.append(f"{str(bin_val)} {str(peak)}\n")
            bin_val += 0.01

        with open(f"{element}_xps_spectrum.txt", "w") as spec:
            spec.writelines(data)

        # Move the spectrum to the run location
        if ctx.obj["RUN_LOC"] != "./":
            os.system(f'mv {element}_xps_spectrum.txt {ctx.obj["RUN_LOC"]}/')

        print("\nplotting spectrum and calculating MABE...")
        Plot.sim_xps_spectrum(
            xps, ctx.obj["RUN_LOC"], ctx.obj["CONSTR_ATOM"], ctx.obj["AT_SPEC"][0], gmp
        )


def projector_wrapper(
    ctx, run_type, occ_type, pbc, l_vecs, spin, ks_range, control_opts
):
    """Force occupation of the Kohn-Sham states."""

    run_loc = ctx.obj["RUN_LOC"]
    geom_inp = ctx.obj["GEOM_INP"]
    control_inp = ctx.obj["CONTROL_INP"]
    atoms = ctx.obj["ATOMS"]
    found_lattice_vecs = ctx.obj["LATTICE_VECS"]
    basis_set = ctx.obj["BASIS_SET"]
    ase = ctx.obj["ASE"]
    nprocs = ctx.obj["NPROCS"]
    binary = ctx.obj["BINARY"]
    hpc = ctx.obj["HPC"]
    constr_atoms = ctx.obj["CONSTR_ATOM"]
    spec_at_constr = ctx.obj["SPEC_AT_CONSTR"]
    spec_mol = ctx.obj["SPEC_MOL"]
    species = ctx.obj["SPECIES"]
    occ = ctx.obj["OCC"]
    print_output = ctx.obj["PRINT"]

    if ase:
        calc = ctx.obj["CALC"]
    else:
        calc = None

    # Used later to redirect STDERR to /dev/null to prevent printing not converged errors
    spec_run_info = None

    # Raise a warning if no additional control options have been specified
    if len(control_opts) < 1 and control_inp is None:
        print(
            "\nWarning: no control options provided, using default options "
            "which can be found in the 'control.in' file"
        )

    # Convert control options to a dictionary
    control_opts = mu.convert_opts_to_dict(control_opts, pbc)

    # Check if the lattice vectors and k_grid have been provided
    if found_lattice_vecs or l_vecs is not None:
        if pbc is None:
            print(
                "Warning: -p/--pbc argument not given, attempting to use"
                " k_grid from control file or previous calculation"
            )

            # Try to parse the k-grid if other calculations have been run
            try:
                pbc_list = []
                for control in glob.glob(f"{run_loc}/**/control.in", recursive=True):
                    with open(control, "r") as control:
                        for line in control:
                            if "k_grid" in line:
                                pbc_list.append(line.split()[1:])

                # If different k_grids have been used for different calculations, then
                # enforce the user to provide the k_grid
                if not pbc_list.count(pbc_list[0]) == len(pbc_list):
                    raise click.MissingParameter(
                        param_hint="-p/--pbc", param_type="option"
                    )
                else:
                    pbc_list = tuple([int(i) for i in pbc_list[0]])
                    control_opts["k_grid"] = pbc_list

            except IndexError:
                raise click.MissingParameter(param_hint="-p/--pbc", param_type="option")

    if run_type == "ground":
        mu.ground_calc(
            run_loc,
            geom_inp,
            control_inp,
            atoms,
            basis_set,
            species,
            calc,
            ase,
            control_opts,
            constr_atoms,
            nprocs,
            binary,
            hpc,
            print_output,
        )

        # Ground must be run separately to hole calculations
        return

    else:  # run_type != ground
        ground_geom = f"{run_loc}/ground/geometry.in"
        ground_control = f"{run_loc}/ground/control.in"

    if len(spec_at_constr) == 0 and constr_atoms is None:
        raise click.MissingParameter(
            param_hint="-c/--constrained_atom or -s/--specific_atom_constraint",
            param_type="option",
        )

    # Convert constr_atoms to a list
    if type(constr_atoms) is not list:
        constr_atoms = [constr_atoms]

    # Create a list of element symbols to constrain
    if len(spec_at_constr) > 0:
        element_symbols = mu.get_element_symbols(ground_geom, spec_at_constr)[0]
        constr_atoms = element_symbols
    else:
        element_symbols = constr_atoms

    fo = ForceOccupation(
        element_symbols,
        run_loc,
        ground_geom,
        control_opts,
        f"{species}/defaults_2020/{basis_set}",
    )

    # Get atom indices from the ground state geometry file
    atom_specifier = fo.read_ground_inp(constr_atoms, spec_at_constr, ground_geom)

    if run_type == "init_1":
        if hpc:
            raise click.BadParameter(
                "the -h/--hpc flag is only supported for the 'hole' run type"
            )

        if len(spec_at_constr) == 0 and len(constr_atoms) == 0:
            raise click.BadParameter(
                "no atoms have been specified to constrain, please use "
                "-c/--constr_atoms or -s/--spec_at_constr options"
            )

        # Check required arguments are given in main()
        mu.check_args(("ks_range", ks_range))

        # TODO allow this for multiple constrained atoms using n_atoms
        for atom in element_symbols:
            fo.get_electronic_structure(atom)

        # Setup files required for the initialisation and hole calculations
        proj = Projector(fo)
        proj.setup_init_1(basis_set, species, ground_control)
        proj.setup_init_2(
            ks_range[0], ks_range[1], occ, occ_type, spin, found_lattice_vecs
        )
        proj.setup_hole(
            ks_range[0], ks_range[1], occ, occ_type, spin, found_lattice_vecs
        )

    spec_run_info = ""

    if run_type == "init_2":
        if hpc:
            raise click.BadParameter(
                "the -h/--hpc flag is only supported for the 'hole' run type"
            )

        if len(spec_at_constr) == 0 and len(constr_atoms) == 0:
            raise click.BadParameter(
                "no atoms have been specified to constrain, please use "
                "-c/--constr_atoms or -s/--spec_at_constr options"
            )

        for i in range(len(atom_specifier)):
            i += 1

            # Catch for if init_1 hasn't been run
            if len(glob.glob(f"{run_loc}/{constr_atoms[0]}{i}/init_1/*restart*")) < 1:
                print(
                    'init_1 restart files not found, please ensure "init_1" has been run'
                )
                raise FileNotFoundError

            if len(control_opts) > 0 or control_inp:
                # Add any additional control options to the init_2 control file
                parsed_control_opts = fo.get_control_keywords(
                    f"{run_loc}/{constr_atoms[0]}{i}/init_2/control.in"
                )
                mod_control_opts = fo.mod_keywords(control_opts, parsed_control_opts)
                control_content = fo.change_control_keywords(
                    f"{run_loc}/{constr_atoms[0]}{i}/init_2/control.in",
                    mod_control_opts,
                )

                with open(
                    f"{run_loc}/{constr_atoms[0]}{i}/init_2/control.in", "w"
                ) as control_file:
                    control_file.writelines(control_content)

            # Copy the restart files to init_2 from init_1
            os.path.isfile(
                glob.glob(f"{run_loc}/{constr_atoms[0]}{i}/init_1/*restart*")[0]
            )
            os.system(
                f"cp {run_loc}/{constr_atoms[0]}{i}/init_1/*restart* {run_loc}/{constr_atoms[0]}{i}/init_2/"
            )

        # Prevent SCF not converged errors from printing
        spec_run_info = " 2>/dev/null"

    if run_type == "hole":
        if len(spec_at_constr) == 0 and len(constr_atoms) == 0:
            raise click.BadParameter(
                "no atoms have been specified to constrain, please use the "
                "-c/--constr_atoms or -s/--spec_at_constr options"
            )

        if hpc:
            mu.check_args(("ks_range", ks_range))

            for atom in element_symbols:
                fo.get_electronic_structure(atom)

            # Setup files required for the initialisation and hole calculations
            proj = Projector(fo)
            proj.setup_init_1(basis_set, species, ground_control)
            proj.setup_init_2(ks_range[0], ks_range[1], occ, occ_type, spin, pbc)
            proj.setup_hole(ks_range[0], ks_range[1], occ, occ_type, spin, pbc)

        # Add molecule identifier to hole geometry.in
        with open(
            f"{run_loc}/{constr_atoms[0]}{atom_specifier[0]}/hole/geometry.in", "r"
        ) as hole_geom:
            lines = hole_geom.readlines()

        lines.insert(4, f"# {spec_mol}\n")

        with open(
            f"{run_loc}/{constr_atoms[0]}{atom_specifier[0]}/hole/geometry.in", "w"
        ) as hole_geom:
            hole_geom.writelines(lines)

        if hpc:
            return

        # Catch for if init_2 hasn't been run
        for i in range(len(atom_specifier)):
            i += 1
            if (
                os.path.isfile(
                    glob.glob(f"{run_loc}/{constr_atoms[0]}{i}/init_2/*restart*")[0]
                )
                is False
            ):
                print(
                    'init_2 restart files not found, please ensure "init_2" has been run'
                )
                raise FileNotFoundError

            if len(control_opts) > 0 or control_inp:
                # Add any additional control options to the hole control file
                parsed_control_opts = fo.get_control_keywords(
                    f"{run_loc}/{constr_atoms[0]}{i}/hole/control.in"
                )
                mod_control_opts = fo.mod_keywords(control_opts, parsed_control_opts)
                control_content = fo.change_control_keywords(
                    f"{run_loc}/{constr_atoms[0]}{i}/hole/control.in", mod_control_opts
                )

                with open(
                    f"{run_loc}/{constr_atoms[0]}{i}/hole/control.in", "w"
                ) as control_file:
                    control_file.writelines(control_content)

            # Copy the restart files to hole from init_2
            os.path.isfile(
                glob.glob(f"{run_loc}/{constr_atoms[0]}{i}/init_2/*restart*")[0]
            )
            os.system(
                f"cp {run_loc}/{constr_atoms[0]}{i}/init_2/*restart* {run_loc}/{constr_atoms[0]}{i}/hole/"
            )

        spec_run_info = ""

    # Run the calculation with a nice progress bar if not already run
    if (
        run_type != "ground"
        and os.path.isfile(
            f"{run_loc}/{constr_atoms[0]}{atom_specifier[0]}/{run_type}/aims.out"
        )
        is False
        and not hpc
    ):
        # Ensure that aims always runs with the following environment variables:
        os.system("export OMP_NUM_THREADS=1")
        os.system("export MKL_NUM_THREADS=1")
        os.system("export MKL_DYNAMIC=FALSE")
        os.system("ulimit -s unlimited")

        if print_output:  # Print live output of calculation
            for i in range(len(atom_specifier)):
                i += 1
                os.system(
                    f"cd {run_loc}/{constr_atoms[0]}{i}/{run_type} && mpirun -n "
                    f"{nprocs} {binary} | tee aims.out {spec_run_info}"
                )

        else:
            with click.progressbar(
                range(len(atom_specifier)), label=f"calculating {run_type}:"
            ) as prog_bar:
                for i in prog_bar:
                    i += 1
                    os.system(
                        f"cd {run_loc}/{constr_atoms[0]}{i}/{run_type} && mpirun -n "
                        f"{nprocs} {binary} > aims.out {spec_run_info}"
                    )

        # print(f"{run_type} calculations completed successfully")

    elif run_type != "ground" and not hpc:
        print(f"{run_type} calculations already completed, skipping calculation...")

    # These need to be passed to process()
    ctx.obj["RUN_TYPE"] = run_type
    ctx.obj["AT_SPEC"] = atom_specifier

    # Compute the dscf energies and plot if option provided
    if run_type == "hole":
        process(ctx)


def basis_wrapper(
    ctx,
    run_type,
    atom_index,
    occ_type,
    multiplicity,
    n_qn,
    l_qn,
    m_qn,
    ks_max,
    control_opts,
):
    """Force occupation of the basis states."""

    # It gets annoying to type the full context object out every time
    run_loc = ctx.obj["RUN_LOC"]
    geom = ctx.obj["GEOM_INP"]
    control = ctx.obj["CONTROL_INP"]
    spec_mol = ctx.obj["SPEC_MOL"]
    atoms = ctx.obj["ATOMS"]
    ase = ctx.obj["ASE"]
    species = ctx.obj["SPECIES"]
    basis_set = ctx.obj["BASIS_SET"]
    nprocs = ctx.obj["NPROCS"]
    binary = ctx.obj["BINARY"]
    hpc = ctx.obj["HPC"]
    constr_atoms = ctx.obj["CONSTR_ATOM"]
    spec_at_constr = ctx.obj["SPEC_AT_CONSTR"]
    occ = ctx.obj["OCC"]
    print_output = ctx.obj["PRINT"]

    if ase:
        calc = ctx.obj["CALC"]
    else:
        calc = None

    # Raise a warning if no additional control options have been specified
    if len(control_opts) < 1 and control is None:
        print(
            "\nWarning: no control options provided, using default options "
            "which can be found in the 'control.in' file"
        )

    # Convert control options to a dictionary
    control_opts = mu.convert_opts_to_dict(control_opts, None)

    if run_type == "ground":
        mu.ground_calc(
            run_loc,
            geom,
            control,
            atoms,
            basis_set,
            species,
            calc,
            ase,
            control_opts,
            constr_atoms,
            nprocs,
            binary,
            hpc,
        )

        # Ground must be run separately to hole calculations
        return

    else:  # run_type == 'hole'
        ground_geom = f"{run_loc}/ground/geometry.in"

    if len(spec_at_constr) == 0 and constr_atoms is None:
        raise click.MissingParameter(
            "No atoms have been specified to constrain, please provide either"
            " the -c/--constrained_atom or the -s/--specific_atom_constraint arguments",
            param_type="option",
        )

    if run_type == "hole":
        mu.check_args(
            ("atom_index", atom_index),
            ("ks_max", ks_max),
            ("n_qn", n_qn),
            ("l_qn", l_qn),
            ("m_qn", m_qn),
        )

        if len(spec_at_constr) == 0 and len(constr_atoms) == 0:
            raise click.BadParameter(
                "no atoms have been specified to constrain, please use the "
                "-c/--constr_atoms or -s/--spec_at_constr options"
            )

        # Ensure that aims always runs with the following environment variables:
        os.system("export OMP_NUM_THREADS=1")
        os.system("export MKL_NUM_THREADS=1")
        os.system("export MKL_DYNAMIC=FALSE")
        os.system("ulimit -s unlimited")

        if os.path.isfile(f"{run_loc}/ground/aims.out") is False:
            raise FileNotFoundError(
                "\nERROR: ground aims.out not found, please ensure the ground calculation has been "
                "run"
            )

        # Convert constr_atoms to a list
        if type(constr_atoms) is not list:
            constr_atoms = [constr_atoms]

        # Create a list of element symbols to constrain
        if len(spec_at_constr) > 0:
            element_symbols = mu.get_element_symbols(ground_geom, spec_at_constr)[0]
            constr_atoms = element_symbols
        else:
            element_symbols = constr_atoms

        # Create the directories required for the hole calculation
        fo = ForceOccupation(
            element_symbols,
            run_loc,
            ground_geom,
            control_opts,
            f"{species}/defaults_2020/{basis_set}",
        )

        # Get atom indices from the ground state geometry file
        atom_specifier = fo.read_ground_inp(constr_atoms, spec_at_constr, ground_geom)

        if (
            os.path.isfile(f"{run_loc}/{constr_atoms[0]}{atom_specifier[0]}/aims.out")
            is True
        ):
            print("hole calculations already completed, skipping calculation...")

        basis = Basis(fo)
        basis.setup_basis(multiplicity, n_qn, l_qn, m_qn, occ, ks_max, occ_type)

        # Add molecule identifier to hole geometry.in
        with open(
            f"{run_loc}{constr_atoms[0]}{atom_specifier[0]}/geometry.in", "r"
        ) as hole_geom:
            lines = hole_geom.readlines()

        lines.insert(4, f"# {spec_mol}\n")

        with open(
            f"{run_loc}/{constr_atoms[0]}{atom_specifier[0]}/geometry.in", "w"
        ) as hole_geom:
            hole_geom.writelines(lines)

        # TODO allow multiple constraints using n_atoms

        for i in range(len(atom_specifier)):
            i += 1

            if len(control_opts) > 0 or control:
                # Add any additional control options to the hole control file
                parsed_control_opts = fo.get_control_keywords(
                    f"{run_loc}/{constr_atoms[0]}{i}/control.in"
                )
                mod_control_opts = fo.mod_keywords(control_opts, parsed_control_opts)
                control_content = fo.change_control_keywords(
                    f"{run_loc}/{constr_atoms[0]}{i}/control.in", mod_control_opts
                )

                with open(
                    f"{run_loc}/{constr_atoms[0]}{i}/control.in", "w"
                ) as control_file:
                    control_file.writelines(control_content)

        if not hpc:  # Run the hole calculation
            if print_output:  # Print live output of calculation
                for i in range(len(atom_specifier)):
                    i += 1
                    os.system(
                        f"cd {run_loc}/{constr_atoms[0]}{i} && mpirun -n "
                        f"{nprocs} {binary} | tee aims.out"
                    )

            else:
                with click.progressbar(
                    range(len(atom_specifier)), label="calculating basis hole:"
                ) as prog_bar:
                    for i in prog_bar:
                        i += 1
                        os.system(
                            f"cd {run_loc}/{constr_atoms[0]}{i} && mpirun -n "
                            f"{nprocs} {binary} > aims.out"
                        )

        # These need to be passed to process()
        ctx.obj["RUN_TYPE"] = run_type
        ctx.obj["AT_SPEC"] = atom_specifier

    # Compute the dscf energies and plot if option provided
    if run_type == "hole":
        process(ctx)
