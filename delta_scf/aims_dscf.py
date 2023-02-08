#!/usr/bin/env python3

import os
import sys
from pathlib import Path

import click
from ase import Atoms
from ase.atoms import default
from ase.io import read, write
from utils.custom_click import MutuallyExclusive as me
from utils.main_utils import MainUtils as mu

from delta_scf.calc_dscf import CalcDeltaSCF as cds
from delta_scf.force_occupation import Basis, ForceOccupation, Projector
from delta_scf.peak_broaden import dos_binning
from delta_scf.plot import Plot


@click.group()
@click.option(
    "-h",
    "--hpc",
    cls=me,
    mutually_exclusive=["--binary"],
    is_flag=True,
    help="setup a calculation primarily for use on a HPC cluster WITHOUT "
    "running the calculation",
)
@click.option(
    "-m",
    "--molecule",
    "spec_mol",
    cls=me,
    mutually_exclusive=["--geometry_input"],
    type=str,
    help="molecule to be used in the calculation",
)
@click.option(
    "-g",
    "--geometry_input",
    cls=me,
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
    cls=me,
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
    "-c",
    "--constrained_atom",
    "constr_atom",
    cls=me,
    mutually_exclusive=["spec_at_constr"],
    type=str,
    help="atom to constrain; constrain all atoms of this element",
)
@click.option(
    "-s",
    "--specific_atom_constraint",
    "spec_at_constr",
    cls=me,
    mutually_exclusive=["constr_atom"],
    multiple=True,
    type=click.IntRange(min=1, max_open=True),
    help="specify specific atoms to constrain by their index in a geometry file",
)
@click.option(
    "-o",
    "--occupation",
    type=float,
    default=0.0,
    show_default=True,
    help="occupation value of the core ",
)
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
    spec_at_constr,
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

    # A geometry file must be given specific atom indices are to be constrained
    if spec_at_constr is not None:
        if not geometry_input:
            raise click.MissingParameter(
                param_hint="'--geometry_input'", param_type="option"
            )

    # Use ASE unless both a custom geometry.in and control.in are specified
    # Also don't use ASE if a control.in is specified
    ase = True
    if geometry_input is not None:
        mu.check_geom_constraints(geometry_input)
        ctx.obj["GEOM"] = geometry_input
    else:
        ctx.obj["GEOM"] = None

    if control_input is not None:
        ase = False
        ctx.obj["CONTROL"] = control_input
    else:
        ctx.obj["CONTROL"] = None

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
                    param_hint="'--molecule' or '--geometry_input'", param_type="option"
                )

    elif "--help" not in sys.argv and ase:
        if spec_mol is not None:
            atoms = mu.build_geometry(spec_mol)
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
                "ensure the FHI-aims binary is in the 'build' directory of the FHI-aims"
                " source code directory, and that the 'species_defaults' directory exists"
            )
            raise NotADirectoryError(
                f"species_defaults directory not found in {Path(binary).parent.parent}"
            )

        # Create the ASE calculator
        if ase:
            aims_calc = mu.create_calc(nprocs, binary, species)
            atoms.calc = aims_calc
            ctx.obj["ATOMS"] = atoms
            ctx.obj["CALC"] = aims_calc

        # User specified context objects
        ctx.obj["SPEC_MOL"] = spec_mol
        ctx.obj["BINARY"] = binary
        ctx.obj["RUN_LOC"] = run_location
        ctx.obj["CONSTR_ATOM"] = constr_atom
        ctx.obj["SPEC_AT_CONSTR"] = spec_at_constr
        ctx.obj["OCC"] = occupation
        ctx.obj["N_ATOMS"] = n_atoms
        ctx.obj["GRAPH"] = graph
        ctx.obj["GMP"] = graph_min_percent
        ctx.obj["NPROCS"] = nprocs
        ctx.obj["DEBUG"] = debug
        ctx.obj["HPC"] = hpc

        # Context objects created in main()
        ctx.obj["SPECIES"] = species
        ctx.obj["ASE"] = ase


def process(ctx):
    """Calculate DSCF values and plot the simulated XPS spectra."""

    # Calculate the delta scf energy and plot
    if ctx.obj["RUN_TYPE"] == "hole":
        grenrgys = cds.read_ground(ctx.obj["RUN_LOC"])
        element, excienrgys = cds.read_atoms(
            ctx.obj["RUN_LOC"], ctx.obj["CONSTR_ATOM"], cds.contains_number
        )
        xps = cds.calc_delta_scf(element, grenrgys, excienrgys)

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
            for xi, yi in zip(x, y):
                dat.append(str(xi) + " " + str(yi) + "\n")

            with open(f"{element}_xps_spectrum.txt", "w") as spec:
                spec.writelines(dat)

            os.system(f'mv {element}_xps_spectrum.txt ./{ctx.obj["RUN_LOC"]}/')

            print("\nplotting spectrum and calculating MABE...")
            Plot.sim_xps_spectrum(
                ctx.obj["RUN_LOC"], ctx.obj["CONSTR_ATOM"], ctx.obj["GMP"]
            )


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
    "-s",
    "--basis_set",
    default="tight",
    show_default=True,
    type=click.Choice(["light", "intermediate", "tight", "really_tight"]),
    help="the basis set to use for the calculation",
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
def projector(ctx, run_type, occ_type, basis_set, pbc, ks_start, ks_stop):
    """Force occupation of the Kohn-Sham states."""

    run_loc = ctx.obj["RUN_LOC"]
    geom = ctx.obj["GEOM"]
    control = ctx.obj["CONTROL"]
    atoms = ctx.obj["ATOMS"]
    ase = ctx.obj["ASE"]
    control_opts = ctx.obj["CONTROL_OPTIONS"]
    nprocs = ctx.obj["NPROCS"]
    binary = ctx.obj["BINARY"]
    hpc = ctx.obj["HPC"]
    constr_atoms = ctx.obj["CONSTR_ATOM"]
    spec_at_constr = ctx.obj["SPEC_AT_CONSTR"]
    n_atoms = ctx.obj["N_ATOMS"]
    species = ctx.obj["SPECIES"]

    # Used later to redirect STDERR to /dev/null to prevent printing not converged errors
    spec_run_info = None

    if pbc == True:
        atoms.set_pbc(True)

    if run_type == "ground":
        mu.ground_calc(
            run_loc, geom, control, atoms, ase, control_opts, nprocs, binary, hpc
        )

    # Create a list of element symbols to constrain
    if spec_at_constr is not None:
        element_symbols = mu.get_element_symbols(geom, spec_at_constr)
    else:
        element_symbols = constr_atoms

    # Makes following code simpler if everything is assumed to be a list
    if type(constr_atoms) is not list:
        list_constr_atoms = list(constr_atoms)
    else:
        list_constr_atoms = constr_atoms

    fo = ForceOccupation(
        list_constr_atoms, spec_at_constr, element_symbols, geom, control
    )

    if run_type == "init_1":
        # Check required arguments are given in main()
        mu.check_args(list_constr_atoms, n_atoms, occ_type, ks_start, ks_stop)

        # Get atom indices from the ground state geometry file
        fo.read_ground_inp("run_dir/ground/geometry.in")

        atom_indices = []
        valencies = []

        # Obtain specified atom valence structures and
        for atom in element_symbols:
            atom_index, valence = fo.get_electronic_structure(atom)
            atom_indices.append(atom_index)
            valencies.append(valence)

        nucleus, n_index, valence_index = Projector.setup_init_1(
            basis_set,
            species,
            constr_atom,
            read_atoms,
            "./run_dir/",
            at_num,
            valence,
        )
        Projector.setup_init_2(
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
        Projector.setup_hole(
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

    # It gets annoying to type the full context object out every time
    run_loc = ctx.obj["RUN_LOC"]
    geom = ctx.obj["GEOM"]
    control = ctx.obj["CONTROL"]
    spec_mol = ctx.obj["SPEC_MOL"]
    atoms = ctx.obj["ATOMS"]
    ase = ctx.obj["ASE"]
    nprocs = ctx.obj["NPROCS"]
    binary = ctx.obj["BINARY"]
    constr_atom = ctx.obj["CONSTR_ATOM"]
    n_atoms = ctx.obj["N_ATOMS"]
    occ = ctx.obj["OCC"]
    hpc = ctx.obj["HPC"]

    mu.check_args(run_loc)

    if run_type == "ground":
        mu.ground_calc(
            run_loc, geom, control, atoms, ase, control_opt, nprocs, binary, hpc
        )

    if (
        run_type == "hole"
        and os.path.isfile(f"{run_loc}/{constr_atom}1/hole/aims.out") is False
    ):
        # Check required arguments are given for main()
        mu.check_args(
            ("spec_mol", spec_mol),
            ("constr_atom", constr_atom),
            ("n_atoms", n_atoms),
            ("occ_type", occ_type),
            ("ks_max", ks_max),
        )

        if os.path.isfile(f"{run_loc}/ground/aims.out") == False:
            print(
                "ground aims.out not found, please ensure the ground calculation has been run"
            )
            raise FileNotFoundError

        if geom or control:
            print(
                "WARNING: custom geometry.in and control.in files will be ignored for hole "
                "runs"
            )

        # Create the directories required for the hole calculation
        Basis.setup_basis(
            constr_atom,
            n_atoms,
            occ,
            ks_max,
            occ_type,
            run_loc,
            control_opt,
        )

        # Add molecule identifier to hole geometry.in
        with open(f"{run_loc}/{constr_atom}1/hole/geometry.in", "r") as hole_geom:
            lines = hole_geom.readlines()

        lines.insert(4, f"# {spec_mol}\n")

        with open(f"{run_loc}/{constr_atom}1/hole/geometry.in", "w") as hole_geom:
            hole_geom.writelines(lines)

        # Run the hole calculation
        with click.progressbar(
            range(1, n_atoms + 1), label="calculating basis hole:"
        ) as prog_bar:
            for i in prog_bar:
                os.system(
                    f"cd {run_loc}/{constr_atom}{i}/hole/ && mpirun -n "
                    f"{nprocs} {binary} > aims.out"
                )

    elif os.path.isfile(f"{run_loc}/{constr_atom}1/hole/aims.out") is True:
        print("hole calculations already completed, skipping calculation...")

    # This needs to be passed to process()
    ctx.obj["RUN_TYPE"] = run_type

    # Compute the dscf energies and plot if option provided
    process(ctx)


if __name__ == "__main__":
    main()
