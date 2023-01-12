#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import click
from ase.io import read
from utils.click_meo import MutuallyExclusiveOption as meo
from utils.main_utils import (build_geometry, check_args,
                              check_geom_constraints, create_calc)

from delta_scf.calc_dscf import *
from delta_scf.force_occupation import *
from delta_scf.peak_broaden import dos_binning
from delta_scf.plot import sim_xps_spectrum


@click.group()
@click.option(
    "-h",
    "--hpc",
    cls=meo,
    mutually_exclusive=["--binary"],
    is_flag=True,
    help="setup a calculation primarily for use on a HPC cluster WITHOUT "
    "running the calculation",
)
@click.option(
    "-m",
    "--molecule",
    "spec_mol",
    cls=meo,
    mutually_exclusive=["--geometry_input"],
    type=str,
    help="molecule to be used in the calculation",
)
@click.option(
    "-g",
    "--geometry_input",
    cls=meo,
    mutually_exclusive=["--molecule"],
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="specify a custom geometry.in instead of using a structure from PubChem or ASE",
)
@click.option(
    "-i",
    "--control_input",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    help="specify a custom control.in instead of automatically generating one",
)
@click.option(
    "-b",
    "--binary",
    cls=meo,
    mutually_exclusive=["hpc"],
    is_flag=True,
    help="modify the path to the FHI-aims binary",
)
@click.option(
    "-r",
    "--run_location",
    default="./",
    show_default=True,
    type=click.Path(file_okay=False, dir_okay=True),
    help="Optionally specify a custom location to run the calculation",
)
@click.option(
    "-c", "--constrained_atom", "constr_atom", type=str, help="atom to constrain"
)
@click.option(
    "-o",
    "--occupation",
    type=float,
    default=0.0,
    show_default=True,
    help="occupation value of the core ",
)
@click.option("-a", "--n_atoms", type=int, help="the number of atoms to constrain")
@click.option("-g", "--graph", is_flag=True, help="print out the simulated XPS spectra")
@click.option(
    "--graph_min_percent",
    default=0.003,
    show_default=True,
    type=float,
    help="specify a value to customise the minimum plotting intensity of the simulated"
    " XPS spectra as a percentage of the maximum intensity",
)
@click.option(
    "-n",
    "--nprocs",
    default=4,
    show_default=True,
    type=int,
    help="number of processors to use",
)
@click.option(
    "-d", "--debug", is_flag=True, help="for developer use: print debug information"
)
@click.pass_context
def main(
    ctx,
    hpc,
    geometry_input,
    control_input,
    binary,
    run_location,
    spec_mol,
    constr_atom,
    occupation,
    n_atoms,
    graph,
    graph_min_percent,
    nprocs,
    debug,
):
    """An interface to automate core-hole constrained occupation methods in
    FHI-aims.

    There is functionality to use both the older and soon-to-be deprecated
    forced_occupation_basis and forced_occupation_projector methods, as well as
    the newer and faster deltascf_basis and deltascf_projector methods. This was
    originally written as a testing application and has since developed into an
    application for automating the basis and projector methods.

    Functionality has been included to run on both a local machine, or a HPC
    cluster. Installation is automated using Poetry, and structures from the ASE
    and PubChem databases can be used to generate the geometry.in file, or the
    geometry.in and control.in can be manually created and passed to this program.
    For full documentation, please refer to the README.md.

    Copyright \u00A9 2022-2023, Dylan Morgan dylan.morgan@warwick.ac.uk
    """

    # Pass global options to subcommands
    ctx.ensure_object(dict)

    # Ensure control.in and geometry.in files are given if on HPC
    if hpc:
        if not geometry_input:
            raise click.MissingParameter(
                param_hint="'--geometry_input'", param_type="option"
            )
        if not control_input:
            raise click.MissingParameter(
                param_hint="'--control_input'", param_type="option"
            )

    # It is not currently supported to use a custom geometry or control with ASE
    # TODO Implement it though
    # if geometry_input and not control_input:
    #     raise click.MissingParameter(
    #         param_hint="'--control_input' must also be specified when using "
    #         "'--geometry_input'",
    #         param_type="option",
    #     )
    # elif geometry_input and not control_input:
    #     raise click.MissingParameter(
    #         param_hint="'--geometry_input' must also be specified when using "
    #         "'--control_input'",
    #         param_type="option",
    #     )
    # elif geometry_input and control_input:
    #     # Check if the geometry.in and control.in are in the same directory
    #     if Path(geometry_input).parent != Path(control_input).parent:
    #         raise click.UsageError(
    #             "geometry.in and control.in must be in the same directory"
    #         )
    #     # Set a parameter to easily determine later if ASE will be used for the
    #     # calculation or not
    #     ase = False
    #     # Also create geometry and control objects to pass to basis/projector
    #     ctx.obj["GEOM"] = geometry_input
    #     ctx.obj["CONTROL"] = control_input

    # Use ASE unless both a custom geometry.in and control.in are specified
    ase = True
    if geometry_input is not None:
        ctx.obj["GEOM"] = geometry_input
    else:
        ctx.obj["GEOM"] = None
    if control_input is not None:
        ctx.obj["CONTROL"] = control_input
    else:
        ctx.obj["CONTROL"] = None
    if control_input is not None and geometry_input is not None:
        ase = False
    if control_input is None and geometry_input is None:
        ase = True
        ctx.obj["GEOM"] = None
        ctx.obj["CONTROL"] = None

    # Find the structure if not given
    # Build the structure if given
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
                    param_hint="'--molecule' or '--geometry_input'", param_type="option"
                )

    elif "--help" not in sys.argv and ase:
        if spec_mol is not None:
            atoms = build_geometry(spec_mol)
        if geometry_input is not None:
            atoms = read(geometry_input)

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
            marker = "\n# Enter the path to the FHI-aims binary above this line"
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
                "ensure the FHI-aims binary is in the 'build' directory of the FHI-aims"
                " source code directory, and that the 'species_defaults' directory exists"
            )
            raise NotADirectoryError(
                f"species_defaults directory not found in {Path(binary).parent.parent}"
            )

        # Create the ASE calculator
        if ase:
            # We have to check for constrained atoms in the geometry.in as there is a
            # bug in ASE currently that causes a TypeError if there are any
            # I have included a link to the reference here:
            # https://gitlab.com/ase/ase/-/issues/1158#note_1156014539
            if geometry_input is not None:
                check_geom_constraints()

            aims_calc = create_calc(nprocs, binary, species)
            atoms.calc = aims_calc
            ctx.obj["ATOMS"] = atoms
            ctx.obj["CALC"] = aims_calc

        # User specified context objects
        ctx.obj["SPEC_MOL"] = spec_mol
        ctx.obj["BINARY"] = binary
        ctx.obj["RUN_LOC"] = run_location
        ctx.obj["CONSTR_ATOM"] = constr_atom
        ctx.obj["OCC"] = occupation
        ctx.obj["N_ATOMS"] = n_atoms
        ctx.obj["GRAPH"] = graph
        ctx.obj["GMP"] = graph_min_percent
        ctx.obj["NPROCS"] = nprocs
        ctx.obj["DEBUG"] = debug

        # Context objects created in main()
        ctx.obj["SPECIES"] = species
        ctx.obj["ASE"] = ase


def process(ctx):
    """Calculate DSCF values and plot the simulated XPS spectra."""

    # Calculate the delta scf energy and plot
    if ctx.obj["RUN_TYPE"] == "hole":
        grenrgys = read_ground(ctx.obj["RUN_LOC"])
        element, excienrgys = read_atoms(
            ctx.obj["RUN_LOC"], ctx.obj["CONSTR_ATOM"], contains_number
        )
        xps = calc_delta_scf(element, grenrgys, excienrgys)

        if ctx.obj["RUN_LOC"] != "./":
            os.system(f"mv {element}_xps_peaks.txt {ctx.obj['RUN_LOC']}")

        if ctx.obj["GRAPH"]:
            # Define parameters for broadening
            xstart = 1
            xstop = 1000
            broad1 = 0.7
            broad2 = 0.7
            firstpeak = 285.0
            ewid1 = firstpeak + 1.0
            ewid2 = firstpeak + 2.0
            mix1 = 0.3
            mix2 = 0.3

            # Apply the broadening
            x, y = dos_binning(
                xps,
                broadening=broad1,
                mix1=mix1,
                mix2=mix2,
                start=xstart,
                stop=xstop,
                coeffs=None,
                broadening2=broad2,
                ewid1=ewid1,
                ewid2=ewid2,
            )

            # Write out the spectrum to a text file
            dat = []
            for (xi, yi) in zip(x, y):
                dat.append(str(xi) + " " + str(yi) + "\n")

            with open(f"{element}_xps_spectrum.txt", "w") as spec:
                spec.writelines(dat)

            os.system(f'mv {element}_xps_spectrum.txt ./{ctx.obj["RUN_LOC"]}/')

            print("\nplotting spectrum and calculating MABE...")
            sim_xps_spectrum(ctx.obj["RUN_LOC"], ctx.obj["CONSTR_ATOM"], ctx.obj["GMP"])


@main.command()
@click.option(
    "-r",
    "--run_type",
    required=True,
    type=click.Choice(["ground", "init_1", "init_2", "hole"]),
    help="the type of calculation to perform",
)
@click.option(
    "-t",
    "--occ_type",
    type=click.Choice(["deltascf_projector", "force_occupation_projector"]),
    help="select whether the old or new occupation routine is used",
)
@click.option(
    "-p", "--pbc", is_flag=True, help="create a cell with periodic boundary conditions"
)
@click.option(
    "-b",
    "--ks_start",
    type=click.IntRange(1),
    help="first Kohn-Sham state to constrain",
)
@click.option(
    "-e",
    "--ks_stop",
    type=click.IntRange(1),
    help="last Kohn-Sham state to constrain",
)
@click.pass_context
def projector(ctx, run_type, occ_type, pbc, ks_start, ks_stop):
    """Force occupation of the Kohn-Sham states."""

    # TODO Update this function with the latest options added to main
    print(
        "The projector command is currently still under development and not ready for "
        "use"
    )
    sys.exit()

    # Used later to redirect STDERR to /dev/null to prevent printing not converged errors
    spec_run_info = None

    if pbc == True:
        ctx.obj["ATOMS"].set_pbc(True)

    if run_type == "ground":
        # Check required arguments are given in main()
        check_args(ctx.obj["SPEC_MOL"])

        # Create the ground directory if it doesn't already exist
        os.system(f"mkdir -p {ctx.obj['RUN_LOC']}/ground")

        # Attach the calculator to the atoms object
        ctx.obj["ATOMS"].calc = ctx.obj["CALC"]

        if os.path.isfile(f"{ctx.obj['RUN_LOC']}/ground/aims.out") == False:
            # Run the ground state calculation
            print("running calculation...")
            ctx.obj["ATOMS"].get_potential_energy()
            print("ground calculation completed successfully")

            # Move files to ground directory
            os.system(
                "mv geometry.in control.in aims.out parameters.ase run_dir/ground/"
            )
        else:
            print("aims.out file found in ground calculation directory")
            print("skipping calculation...")

    if run_type == "init_1":
        # Check required arguments are given in main()
        check_args(
            ctx.obj["CONSTR_ATOM"], ctx.obj["N_ATOMS"], occ_type, ks_start, ks_stop
        )

        basis_set = "tight"
        element_symbols, read_atoms = read_ground_inp(
            ctx.obj["CONSTR_ATOM"], "run_dir/ground/geometry.in"
        )
        at_num, valence = get_electronic_structure(
            element_symbols, ctx.obj["CONSTR_ATOM"]
        )
        nucleus, n_index, valence_index = setup_init_1(
            basis_set,
            ctx.obj["SPECIES"],
            ctx.obj["CONSTR_ATOM"],
            read_atoms,
            "./run_dir/",
            at_num,
            valence,
        )
        setup_init_2(
            [i for i in range(ks_start + 1, ks_stop + 1)],
            "./run_dir/",
            ctx.obj["CONSTR_ATOM"],
            ctx.obj["N_ATOMS"],
            at_num,
            valence,
            n_index,
            valence_index,
            occ_type,
        )
        setup_hole(
            "./run_dir/",
            [i for i in range(ks_start + 1, ks_stop + 1)],
            ctx.obj["CONSTR_ATOM"],
            ctx.obj["N_ATOMS"],
            nucleus,
            valence,
            n_index,
            valence_index,
        )

        spec_run_info = ""

    if run_type == "init_2":
        # Check required arguments are given in main()
        check_args(ctx.obj["CONSTR_ATOM"], ctx.obj["N_ATOMS"])

        # Catch for if init_1 hasn't been run
        for i in range(1, ctx.obj["N_ATOMS"] + 1):
            if (
                os.path.isfile(
                    f"run_dir/{ctx.obj['CONSTR_ATOM']}{i}/init_1/restart_file"
                )
                is False
            ):
                print(
                    'init_1 restart files not found, please ensure "init_1" has been run'
                )
                raise FileNotFoundError

        # Move the restart files to init_1
        for i in range(1, ctx.obj["N_ATOMS"] + 1):
            os.path.isfile(f"run_dir/{ctx.obj['CONSTR_ATOM']}{i}/init_1/restart_file")
            os.system(
                f"cp run_dir/{ctx.obj['CONSTR_ATOM']}{i}/init_1/restart* run_dir/{ctx.obj['CONSTR_ATOM']}{i}/init_2/"
            )

        # Prevent SCF not converged errors from printing
        spec_run_info = " 2>/dev/null"

    if run_type == "hole":
        # Check required arguments are given in main()
        check_args(ctx.obj["SPEC_MOL"], ctx.obj["CONSTR_ATOM"], ctx.obj["N_ATOMS"])

        # Add molecule identifier to hole geometry.in
        with open(
            f"run_dir/{ctx.obj['CONSTR_ATOM']}1/hole/geometry.in", "r"
        ) as hole_geom:
            lines = hole_geom.readlines()

        lines.insert(4, f"# {ctx.obj['SPEC_MOL']}\n")

        with open(
            f"run_dir/{ctx.obj['CONSTR_ATOM']}1/hole/geometry.in", "w"
        ) as hole_geom:
            hole_geom.writelines(lines)

        # Catch for if init_2 hasn't been run
        for i in range(1, ctx.obj["N_ATOMS"] + 1):
            if (
                os.path.isfile(
                    f"run_dir/{ctx.obj['CONSTR_ATOM']}{i}/init_2/restart_file"
                )
                is False
            ):
                print(
                    'init_2 restart files not found, please ensure "init_2" has been run'
                )
                raise FileNotFoundError

        # Move the restart files to hole
        for i in range(1, ctx.obj["N_ATOMS"] + 1):
            os.path.isfile(f"run_dir/{ctx.obj['CONSTR_ATOM']}{i}/init_2/restart_file")
            os.system(
                f"cp run_dir/{ctx.obj['CONSTR_ATOM']}{i}/init_2/restart* run_dir/{ctx.obj['CONSTR_ATOM']}{i}/hole/"
            )

        spec_run_info = ""

    # Run the calculation with a nice progress bar if not already run
    if (
        run_type != "ground"
        and os.path.isfile(f"run_dir/{ctx.obj['CONSTR_ATOM']}1/{run_type}/aims.out")
        == False
    ):
        with click.progressbar(
            range(1, ctx.obj["N_ATOMS"] + 1), label=f"calculating {run_type}:"
        ) as bar:
            for i in bar:
                os.system(
                    f"cd ./run_dir/{ctx.obj['CONSTR_ATOM']}{i}/{run_type} && mpirun -n {ctx.obj['PROCS']} {ctx.obj['BINARY']} > aims.out{spec_run_info}"
                )

        print(f"{run_type} calculations completed successfully")

    elif run_type != "ground":
        print(f"{run_type} calculations already completed, skipping calculation...")

    # Compute the dscf energies and plot if option provided
    process(ctx)


@main.command()
@click.option(
    "-r",
    "--run_type",
    required=True,
    type=click.Choice(["ground", "hole"]),
    help="select the type of calculation to perform",
)
@click.option(
    "-t",
    "--occ_type",
    type=click.Choice(["deltascf_basis", "force_occupation_basis"]),
    help="select whether the old or new occupation routine is used",
)
@click.option(
    "-e",
    "--ks_max",
    type=click.IntRange(1),
    help="maximum Kohn-Sham state to constrain",
)
@click.option(
    "-c",
    "--control_opt",
    multiple=True,
    type=str,
    help="provide additional options to be used in 'control.in'",
)
@click.pass_context
def basis(ctx, run_type, occ_type, ks_max, control_opt):
    """Force occupation of the basis states."""

    run_loc = ctx.obj["RUN_LOC"]
    check_args(ctx.obj["RUN_LOC"])

    if run_type == "ground":
        os.system(f"mkdir -p {run_loc}/ground")

        # Create the ground directory if it doesn't already exist
        if ctx.obj["GEOM"] is not None and ctx.obj["CONTROL"] is not None:
            os.system(f"mv {ctx.obj['GEOM']} {ctx.obj['RUN_LOC']} {run_loc}/ground")
        else:
            # Check required arguments are given for main()
            check_args(("spec_mol", ctx.obj["SPEC_MOL"]))

        if os.path.isfile(f"{run_loc}/ground/aims.out") == False:
            # Run the ground state calculation
            print("running calculation...")

            if ctx.obj["ASE"]:
                if len(control_opt) > 0:
                    print(
                        "WARNING: it is required to use '--control_input' and "
                        "'--geometry_input' instead of supplying additional control "
                        "options for the ground calculations"
                    )

                ctx.obj["ATOMS"].get_potential_energy()
                # Move files to ground directory
                os.system(
                    "mv geometry.in control.in aims.out parameters.ase "
                    f"{run_loc}/ground/"
                )
            else:
                os.system(
                    f'cd {run_loc}/ground && mpirun -n {ctx.obj["NPROCS"]} '
                    f'{ctx.obj["BINARY"]} > aims.out'
                )

            print("ground calculation completed successfully")

        else:
            print("aims.out file found in ground calculation directory")
            print("skipping calculation...")

    if (
        run_type == "hole"
        and os.path.isfile(f"{run_loc}/{ctx.obj['CONSTR_ATOM']}1/hole/aims.out")
        is False
    ):
        # Check required arguments are given for main()
        check_args(
            ("spec_mol", ctx.obj["SPEC_MOL"]),
            ("constr_atom", ctx.obj["CONSTR_ATOM"]),
            ("n_atoms", ctx.obj["N_ATOMS"]),
            ("occ_type", occ_type),
            ("ks_max", ks_max),
        )

        if os.path.isfile(f"{run_loc}/ground/aims.out") == False:
            print(
                "ground aims.out not found, please ensure the ground calculation has been run"
            )
            raise FileNotFoundError

        if ctx.obj["GEOM"] or ctx.obj["CONTROL"]:
            print(
                "WARNING: custom geometry.in and control.in files will be ignored for hole "
                "runs"
            )

        # Create the directories required for the hole calculation
        setup_fob(
            ctx.obj["CONSTR_ATOM"],
            ctx.obj["N_ATOMS"],
            ctx.obj["OCC"],
            ks_max,
            occ_type,
            run_loc,
            control_opt,
        )

        # Add molecule identifier to hole geometry.in
        with open(
            f"{run_loc}/{ctx.obj['CONSTR_ATOM']}1/hole/geometry.in", "r"
        ) as hole_geom:
            lines = hole_geom.readlines()

        lines.insert(4, f"# {ctx.obj['SPEC_MOL']}\n")

        with open(
            f"{ctx.obj['RUN_LOC']}/{ctx.obj['CONSTR_ATOM']}1/hole/geometry.in", "w"
        ) as hole_geom:
            hole_geom.writelines(lines)

        # Run the hole calculation
        with click.progressbar(
            range(1, ctx.obj["N_ATOMS"] + 1), label="calculating basis hole:"
        ) as prog_bar:
            for i in prog_bar:
                os.system(
                    f"cd {run_loc}/{ctx.obj['CONSTR_ATOM']}{i}/hole/ && mpirun -n "
                    f"{ctx.obj['NPROCS']} {ctx.obj['BINARY']} > aims.out"
                )

    elif os.path.isfile(f"{run_loc}/{ctx.obj['CONSTR_ATOM']}1/hole/aims.out") is True:
        print("hole calculations already completed, skipping calculation...")

    # This needs to be passed to process()
    ctx.obj["RUN_TYPE"] = run_type

    # Compute the dscf energies and plot if option provided
    process(ctx)


if __name__ == "__main__":
    main()
