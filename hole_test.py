#!/usr/bin/env python3

import os
import socket
import sys
from pathlib import Path
from types import NoneType

import click
from ase.build import molecule
from ase.calculators.aims import Aims
from ase.data.pubchem import pubchem_atoms_search
from ase.io import read

from calc_dscf import *
from force_occupation import *
from peak_broaden import *
from plot import *


def build_geometry(spec_mol):
    """Check different databases to create a geometry.in"""

    try:
        atoms = molecule(spec_mol)
        print("molecule found in ASE database")
        return atoms
    except KeyError:
        print("molecule not found in ASE database, searching PubChem...")

    try:
        atoms = pubchem_atoms_search(name=spec_mol)
        print("molecule found as a PubChem name")
        return atoms
    except ValueError:
        print(f"{spec_mol} not found in PubChem name")

    try:
        atoms = pubchem_atoms_search(cid=spec_mol)
        print("molecule found in PubChem CID")
        return atoms
    except ValueError:
        print(f"{spec_mol} not found in PubChem CID")

    try:
        atoms = pubchem_atoms_search(smiles=spec_mol)
        print("molecule found in PubChem SMILES")
        return atoms
    except ValueError:
        print(f"{spec_mol} not found in PubChem smiles")
        print(f"{spec_mol} not found in PubChem or ASE database")
        print("aborting...")
        sys.exit(1)


def create_calc(procs, binary, species):
    """Create a calculator object"""

    aims_calc = Aims(
        xc="pbe",
        spin="collinear",
        default_initial_moment=0,
        aims_command=f"mpirun -n {procs} {binary}",
        species_dir=f"{species}defaults_2020/tight/",
    )

    return aims_calc


def check_args(*args):
    """Check if required arguments are specified"""
    def_args = locals()

    for arg in def_args["args"]:
        if arg[1] is None:
            raise click.MissingParameter(
                param_hint=f"'--{arg[0]}'", param_type="option"
            )


@click.group()
@click.option(
    "-m",
    "--molecule",
    "spec_mol",
    type=str,
    help="molecule to be used in the calculation",
)
@click.option(
    "-c", "--constrained_atom", "constr_atom", type=str, help="atom to constrain"
)
@click.option("-a", "--n_atoms", type=int, help="the number of atoms to constrain")
@click.option("-g", "--graph", is_flag=True, help="print out the simulated XPS spectra")
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
def cli(ctx, spec_mol, constr_atom, n_atoms, graph, nprocs, debug):
    """Test constrained occupation methods in FHI-aims."""

    # Set general parameters
    home_dir = str(Path.home())
    hostname = socket.gethostname()
    species = f"{home_dir}/Programming/mac_projects/FHIaims/species_defaults/"
    binary = None
    atoms = None

    # Find the structure if not given
    if spec_mol is None and "--help" not in sys.argv:
        atoms = read("./test_dirs/ground/geometry.in")
        print("molecule argument not provided, defaulting to using geometry.in file")
    elif "--help" not in sys.argv:
        atoms = build_geometry(spec_mol)

    # Set different binaries for different machines
    if hostname == "apollo":
        # Change the binary to use if debugging
        if debug:
            binary = f"{home_dir}/Programming/projects/FHIaims/build/aims_debug.x"
        else:
            binary = f"{home_dir}/Programming/projects/FHIaims/build/aims.221103.scalapack.mpi.x"

    elif hostname == "maccie":
        if debug:
            binary = f"{home_dir}/Programming/mac_projects/FHIaims/build/aims_debug.x"
        else:
            binary = f"{home_dir}/Programming/mac_projects/FHIaims/build/aims.221103.scalapack.mpi.x"

    # Pass global options to subcommands
    ctx.ensure_object(dict)

    # Create the ASE calculator
    if "--help" not in sys.argv:
        aims_calc = create_calc(nprocs, binary, species)
        atoms.calc = aims_calc

        # User specified
        ctx.obj["SPEC_MOL"] = spec_mol
        ctx.obj["CONSTR_ATOM"] = constr_atom
        ctx.obj["N_ATOMS"] = n_atoms
        ctx.obj["GRAPH"] = graph
        ctx.obj["NPROCS"] = nprocs
        ctx.obj["DEBUG"] = debug

        # Created in cli()
        ctx.obj["SPECIES"] = species
        ctx.obj["BINARY"] = binary
        ctx.obj["ATOMS"] = atoms
        ctx.obj["CALC"] = aims_calc


@click.pass_context
def process(ctx):

    # Calculate the delta scf energy and plot
    if ctx.obj["RUN_TYPE"] == "hole":
        grenrgys = read_ground("test_dirs/")
        element, excienrgys = read_atoms(
            "test_dirs/", ctx.obj["CONSTR_ATOM"], contains_number
        )
        xps = calc_delta_scf(element, grenrgys, excienrgys)
        os.system(f"mv {element}_xps_peaks.txt test_dirs/")

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

        os.system(f"mv {element}_xps_spectrum.txt ./test_dirs/")

    if ctx.obj["GRAPH"]:
        print("\nplotting spectrum and calculating MABE...")
        sim_xps_spectrum(ctx.obj["CONSTR_ATOM"])


@cli.command()
@click.option(
    "-t",
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

    # Used later to redirect STDERR to /dev/null to prevent printing not converged errors
    spec_run_info = None

    if pbc == True:
        ctx.obj["ATOMS"].set_pbc(True)

    if run_type == "ground":
        # Check required arguments are given in cli()
        check_args(ctx.obj["SPEC_MOL"])

        # Create the ground directory if it doesn't already exist
        os.system("mkdir -p test_dirs/ground")

        # Attach the calculator to the atoms object
        ctx.obj["ATOMS"].calc = ctx.obj["CALC"]

        if os.path.isfile(f"test_dirs/ground/aims.out") == False:
            # Run the ground state calculation
            print("running calculation...")
            ctx.obj["ATOMS"].get_potential_energy()
            print("ground calculation completed successfully")

            # Move files to ground directory
            os.system(
                "mv geometry.in control.in aims.out parameters.ase test_dirs/ground/"
            )
        else:
            print("aims.out file found in ground calculation directory")
            print("skipping calculation...")

    if run_type == "init_1":
        # Check required arguments are given in cli()
        check_args(
            ctx.obj["CONSTR_ATOM"], ctx.obj["N_ATOMS"], occ_type, ks_start, ks_stop
        )

        basis_set = "tight"
        element_symbols, read_atoms = read_ground_inp(
            ctx.obj["CONSTR_ATOM"], "test_dirs/ground/geometry.in"
        )
        at_num, valence = get_electronic_structure(
            element_symbols, ctx.obj["CONSTR_ATOM"]
        )
        nucleus, n_index, valence_index = setup_init_1(
            basis_set,
            ctx.obj["SPECIES"],
            ctx.obj["CONSTR_ATOM"],
            read_atoms,
            "./test_dirs/",
            at_num,
            valence,
        )
        setup_init_2(
            [i for i in range(ks_start + 1, ks_stop + 1)],
            "./test_dirs/",
            ctx.obj["CONSTR_ATOM"],
            ctx.obj["N_ATOMS"],
            at_num,
            valence,
            n_index,
            valence_index,
            occ_type,
        )
        setup_hole(
            "./test_dirs/",
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
        # Check required arguments are given in cli()
        check_args(ctx.obj["CONSTR_ATOM"], ctx.obj["N_ATOMS"])

        # Catch for if init_1 hasn't been run
        for i in range(1, ctx.obj["N_ATOMS"] + 1):
            if (
                os.path.isfile(
                    f"test_dirs/{ctx.obj['CONSTR_ATOM']}{i}/init_1/restart_file"
                )
                is False
            ):
                print(
                    'init_1 restart files not found, please ensure "init_1" has been run'
                )
                raise FileNotFoundError

        # Move the restart files to init_1
        for i in range(1, ctx.obj["N_ATOMS"] + 1):
            os.path.isfile(f"test_dirs/{ctx.obj['CONSTR_ATOM']}{i}/init_1/restart_file")
            os.system(
                f"cp test_dirs/{ctx.obj['CONSTR_ATOM']}{i}/init_1/restart* test_dirs/{ctx.obj['CONSTR_ATOM']}{i}/init_2/"
            )

        # Prevent SCF not converged errors from printing
        spec_run_info = " 2>/dev/null"

    if run_type == "hole":
        # Check required arguments are given in cli()
        check_args(ctx.obj["SPEC_MOL"], ctx.obj["CONSTR_ATOM"], ctx.obj["N_ATOMS"])

        # Add molecule identifier to hole geometry.in
        with open(
            f"test_dirs/{ctx.obj['CONSTR_ATOM']}1/hole/geometry.in", "r"
        ) as hole_geom:
            lines = hole_geom.readlines()

        lines.insert(4, f"# {ctx.obj['SPEC_MOL']}\n")

        with open(
            f"test_dirs/{ctx.obj['CONSTR_ATOM']}1/hole/geometry.in", "w"
        ) as hole_geom:
            hole_geom.writelines(lines)

        # Catch for if init_2 hasn't been run
        for i in range(1, ctx.obj["N_ATOMS"] + 1):
            if (
                os.path.isfile(
                    f"test_dirs/{ctx.obj['CONSTR_ATOM']}{i}/init_2/restart_file"
                )
                is False
            ):
                print(
                    'init_2 restart files not found, please ensure "init_2" has been run'
                )
                raise FileNotFoundError

        # Move the restart files to hole
        for i in range(1, ctx.obj["N_ATOMS"] + 1):
            os.path.isfile(f"test_dirs/{ctx.obj['CONSTR_ATOM']}{i}/init_2/restart_file")
            os.system(
                f"cp test_dirs/{ctx.obj['CONSTR_ATOM']}{i}/init_2/restart* test_dirs/{ctx.obj['CONSTR_ATOM']}{i}/hole/"
            )

        spec_run_info = ""

    # Run the calculation with a nice progress bar if not already run
    if (
        run_type != "ground"
        and os.path.isfile(f"test_dirs/{ctx.obj['CONSTR_ATOM']}1/{run_type}/aims.out")
        == False
    ):
        with click.progressbar(
            range(1, ctx.obj["N_ATOMS"] + 1), label=f"calculating {run_type}:"
        ) as bar:
            for i in bar:
                os.system(
                    f"cd ./test_dirs/{ctx.obj['CONSTR_ATOM']}{i}/{run_type} && mpirun -n {ctx.obj['PROCS']} {ctx.obj['BINARY']} > aims.out{spec_run_info}"
                )

        print(f"{run_type} calculations completed successfully")

    elif run_type != "ground":
        print(f"{run_type} calculations already completed, skipping calculation...")

    # Compute the dscf energies and plot if option provided
    process()


@cli.command()
@click.option(
    "-r",
    "--run_type",
    required=True,
    type=click.Choice(["ground", "hole"]),
    help="select the type of calculation to perform",
)
@click.option(
    "-o",
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
@click.pass_context
def basis(ctx, run_type, occ_type, ks_max):
    """Force occupation of the basis states."""

    if run_type == "ground":
        # Check required arguments are given for cli()
        check_args(ctx.obj["SPEC_MOL"])

        # Create the ground directory if it doesn't already exist
        os.system("mkdir -p test_dirs/ground")

        if os.path.isfile(f"test_dirs/ground/aims.out") == False:
            # Run the ground state calculation
            print("running calculation...")
            ctx.obj["ATOMS"].get_potential_energy()
            print("ground calculation completed successfully")

            # Move files to ground directory
            os.system(
                "mv geometry.in control.in aims.out parameters.ase test_dirs/ground/"
            )
        else:
            print("aims.out file found in ground calculation directory")
            print("skipping calculation...")

    if (
        run_type == "hole"
        and os.path.isfile(f"test_dirs/{ctx.obj['CONSTR_ATOM']}1/hole/aims.out")
        == False
    ):
        # Check required arguments are given for cli()
        check_args(
            ("spec_mol", ctx.obj["SPEC_MOL"]),
            ("constr_atom", ctx.obj["CONSTR_ATOM"]),
            ("n_atoms", ctx.obj["N_ATOMS"]),
            ("occ_type", occ_type),
            ("ks_max", ks_max),
        )

        if os.path.isfile(f"test_dirs/ground/aims.out") == False:
            print(
                "ground aims.out not found, please ensure the ground calculation has been run"
            )
            raise FileNotFoundError

        setup_fob(ctx.obj["CONSTR_ATOM"], ctx.obj["N_ATOMS"], ks_max, occ_type)

        # Add molecule identifier to hole geometry.in
        with open(
            f"test_dirs/{ctx.obj['CONSTR_ATOM']}1/hole/geometry.in", "r"
        ) as hole_geom:
            lines = hole_geom.readlines()

        lines.insert(4, f"# {ctx.obj['SPEC_MOL']}\n")

        with open(
            f"test_dirs/{ctx.obj['CONSTR_ATOM']}1/hole/geometry.in", "w"
        ) as hole_geom:
            hole_geom.writelines(lines)

        # Run the hole calculation
        with click.progressbar(
            range(1, ctx.obj["N_ATOMS"] + 1), label="calculating basis hole:"
        ) as bar:
            for i in bar:
                os.system(
                    f"cd ./test_dirs/{ctx.obj['CONSTR_ATOM']}{i}/hole/ && mpirun -n {ctx.obj['NPROCS']} {ctx.obj['BINARY']} > aims.out"
                )

    elif os.path.isfile(f"test_dirs/{ctx.obj['CONSTR_ATOM']}1/hole/aims.out") == True:
        print("hole calculations already completed, skipping calculation...")

    # This needs to be passed to main()
    ctx.obj["RUN_TYPE"] = run_type

    # Compute the dscf energies and plot if option provided
    process()


if __name__ == "__main__":
    cli()
