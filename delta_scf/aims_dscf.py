#!/usr/bin/env python3

import glob
import os
import sys
from pathlib import Path

import click
from ase import Atoms
from ase.io import read
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
    "-e",
    "--geometry_input",
    cls=me,
    mutually_exclusive=["--molecule"],
    nargs=1,
    type=click.File(),
    help="specify a custom geometry.in instead of using a structure from PubChem or ASE",
)
@click.option(
    "-i",
    "--control_input",
    nargs=1,
    type=click.File(),
    help="specify a custom control.in instead of automatically generating one",
)
@click.option(
    "-y",
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
    help="occupation value of the core hole",
)
@click.option(
    "-a",
    "--n_atoms",
    type=click.IntRange(1),
    default=1,
    show_default=True,
    help="number of atoms to constrain per calculation",
)
@click.option(
    "-b",
    "--basis_set",
    default="tight",
    show_default=True,
    type=click.Choice(["light", "intermediate", "tight", "really_tight"]),
    help="the basis set to use for the calculation",
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
    basis_set,
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

    # A geometry file must be given if specific atom indices are to be constrained
    if len(spec_at_constr) > 0:
        if not geometry_input:
            raise click.MissingParameter(
                param_hint="'--geometry_input'", param_type="option"
            )

    # Use ASE unless both a custom geometry.in and control.in are specified
    # Also don't use ASE if a control.in is specified
    ase = True
    if geometry_input is not None:
        found_lattice_vecs = mu.check_geom(geometry_input)
        ctx.obj["GEOM_INP"] = geometry_input.name
        ctx.obj["LATTICE_VECS"] = found_lattice_vecs
    else:
        ctx.obj["GEOM_INP"] = None
        ctx.obj["LATTICE_VECS"] = None

    if control_input is not None:
        ase = False
        ctx.obj["CONTROL_INP"] = control_input.name
    else:
        ctx.obj["CONTROL_INP"] = None

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
            atoms = read(geometry_input.name)

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
        ctx.obj["N_ATOMS"] = n_atoms  # TODO
        ctx.obj["BASIS_SET"] = basis_set
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
            broad = 0.7
            firstpeak = 285.0
            ewid1 = firstpeak + 1.0
            ewid2 = firstpeak + 2.0
            mix1 = 0.3
            mix2 = 0.3

            # Apply the broadening
            x, y = dos_binning(
                xps,
                broadening=broad,
                mix1=mix1,
                mix2=mix2,
                start=xstart,
                stop=xstop,
                coeffs=None,
                broadening2=broad,
                ewid1=ewid1,
                ewid2=ewid2,
            )

            # Write out the spectrum to a text file
            dat = []
            for xi, yi in zip(x, y):
                dat.append(f"{str(xi)} {str(yi)}\n")

            with open(f"{element}_xps_spectrum.txt", "w") as spec:
                spec.writelines(dat)

            # Move the spectrum to the run location
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
    default="deltascf_projector",
    show_default=True,
    type=click.Choice(["deltascf_projector", "force_occupation_projector"]),
    help="select whether the old or new occupation routine is used",
)
@click.option(
    "-p",
    "--pbc",
    nargs=3,
    type=int,
    help="give the k-grid for a periodic calculation",
)
@click.option(
    "-l",
    "--lattice_vectors",
    "l_vecs",
    nargs=1,
    type=list,
    help="provide the lattice vectors in a 3x3 matrix",
)
@click.option(
    "-u",
    "--spin",
    type=int,
    default=0,
    show_default=True,
    help="set the spin of the system",
)
@click.option(
    "-k",
    "--ks_range",
    nargs=2,
    type=click.IntRange(1),
    help="range of Kohn-Sham states to constrain taken with 2 arguments",
)
@click.option(
    "-c",
    "--control_opts",
    multiple=True,
    type=str,
    help="provide additional options to be used in 'control.in' in a key=value format",
)
@click.pass_context
def projector(ctx, run_type, occ_type, pbc, l_vecs, spin, ks_range, control_opts):
    """Force occupation of the Kohn-Sham states."""

    run_loc = ctx.obj["RUN_LOC"]
    geom_inp = ctx.obj["GEOM_INP"]
    control_inp = ctx.obj["CONTROL_INP"]
    atoms = ctx.obj["ATOMS"]
    found_lattice_vecs = ctx.obj["LATTICE_VECS"]
    calc = ctx.obj["CALC"]
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

    # Used later to redirect STDERR to /dev/null to prevent printing not converged errors
    spec_run_info = None

    # Raise a warning if no additional control options have been specified
    if len(control_opts) < 1:
        print(
            "\nWARNING: no control options provided, using default options "
            "which can be found in the 'control.in' file"
        )

    # Convert control options to a dictionary
    control_opts = mu.convert_opts_to_dict(control_opts, pbc)

    # Check if the lattice vectors and k_grid have been provided
    if found_lattice_vecs or l_vecs is not None:
        if pbc is None:
            print(
                "WARNING: -p/--pbc argument not given, attempting to use"
                " k_grid from previous calculation"
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
                        "\nERROR: 'k_grid' keyword found in "
                        "control.in but -p/--pbc option has not been provided"
                    )
                else:
                    pbc_list = tuple([int(i) for i in pbc_list[0]])
                    control_opts["k_grid"] = pbc_list

            except FileNotFoundError:
                raise click.MissingParameter(
                    "\nERROR: 'lattice_vector' keyword found in "
                    "geometry.in but -p/--pbc option has not provided been"
                )

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
            nprocs,
            binary,
            hpc,
        )

        # Ground must be run separately to hole calculations
        return

    else:  # run_type != ground
        ground_geom = f"{run_loc}/ground/geometry.in"
        ground_control = f"{run_loc}/ground/control.in"

    if len(spec_at_constr) == 0 and constr_atoms is None:
        raise click.MissingParameter(
            "No atoms have been specified to constrain, please provide either"
            " the -c/--constrained_atom or the -s/--specific_atom_constraint arguments",
            param_type="option",
        )

    # Create a list of element symbols to constrain
    if len(spec_at_constr) > 0:
        element_symbols = mu.get_element_symbols(ground_geom, spec_at_constr)
    else:
        element_symbols = constr_atoms

    # Makes following code simpler if everything is assumed to be a list
    if type(constr_atoms) is not list:
        list_constr_atoms = list(constr_atoms)
    else:
        list_constr_atoms = constr_atoms

    fo = ForceOccupation(
        element_symbols,
        run_loc,
        ground_geom,
        control_opts,
    )

    # Get atom indices from the ground state geometry file
    atom_specifier = fo.read_ground_inp(list_constr_atoms, spec_at_constr, ground_geom)

    if run_type == "init_1":
        if hpc:
            raise click.BadParameter(
                "ERROR: the -h/--hpc flag is only supported for the 'hole' run type"
            )

        if len(spec_at_constr) == 0 and len(constr_atoms) == 0:
            raise click.BadParameter(
                "ERROR: no atoms have been specified to constrain, please use the "
                "-a/--constr_atoms or -s/--spec_at_constr options"
            )

        # Check required arguments are given in main()
        mu.check_args(("ks_range", ks_range))

        # TODO allow this for multiple constrained atoms using n_atoms
        for atom in element_symbols:
            fo.get_electronic_structure(atom)

        # Setup files required for the initialisation and hole calculations
        proj = Projector(fo)
        proj.setup_init_1(basis_set, species, ground_control)
        proj.setup_init_2(ks_range[0], ks_range[1], occ, occ_type, spin)
        proj.setup_hole(ks_range[0], ks_range[1], occ, occ_type, spin)

    spec_run_info = ""

    if run_type == "init_2":
        if hpc:
            raise click.BadParameter(
                "ERROR: the -h/--hpc flag is only supported for the 'hole' run type"
            )

        if len(spec_at_constr) == 0 and len(constr_atoms) == 0:
            raise click.BadParameter(
                "ERROR: no atoms have been specified to constrain, please use the "
                "-a/--constr_atoms or -s/--spec_at_constr options"
            )

        # Catch for if init_1 hasn't been run
        for i in range(len(atom_specifier)):
            i += 1

            if len(glob.glob(f"{run_loc}/{constr_atoms}{i}/init_1/*restart*")) < 1:
                print(
                    'init_1 restart files not found, please ensure "init_1" has been run'
                )
                raise FileNotFoundError

            if len(control_opts) > 0:
                # Add any additional control options to the init_2 control file
                parsed_control_opts = fo.get_control_keywords(
                    f"{run_loc}/{constr_atoms}{i}/init_2/control.in"
                )
                control_opts = fo.mod_keywords(control_opts, parsed_control_opts)
                control_content = fo.change_control_keywords(
                    f"{run_loc}/{constr_atoms}{i}/init_2/control.in", control_opts
                )

                with open(
                    f"{run_loc}/{constr_atoms}{i}/init_2/control.in", "w"
                ) as control_file:
                    control_file.writelines(control_content)

            # Copy the restart files to init_2 from init_1
            os.path.isfile(
                glob.glob(f"{run_loc}/{constr_atoms}{i}/init_1/*restart*")[0]
            )
            os.system(
                f"cp {run_loc}/{constr_atoms}{i}/init_1/*restart* {run_loc}/{constr_atoms}{i}/init_2/"
            )

        # Prevent SCF not converged errors from printing
        spec_run_info = " 2>/dev/null"

    if run_type == "hole":
        if len(spec_at_constr) == 0 and len(constr_atoms) == 0:
            raise click.BadParameter(
                "ERROR: no atoms have been specified to constrain, please use the "
                "-a/--constr_atoms or -s/--spec_at_constr options"
            )

        # Add molecule identifier to hole geometry.in
        if hpc:
            mu.check_args(("ks_range", ks_range))

            # Setup files required for the initialisation and hole calculations
            proj = Projector(fo)
            proj.setup_init_1(basis_set, species, ground_control)
            proj.setup_init_2(ks_range[0], ks_range[1], occ, occ_type, spin)
            proj.setup_hole(ks_range[0], ks_range[1], occ, occ_type, spin)

        with open(f"{run_loc}/{constr_atoms}1/hole/geometry.in", "r") as hole_geom:
            lines = hole_geom.readlines()

        lines.insert(4, f"# {spec_mol}\n")

        with open(f"{run_loc}/{constr_atoms}1/hole/geometry.in", "w") as hole_geom:
            hole_geom.writelines(lines)

        if hpc:
            return

        # Catch for if init_2 hasn't been run
        for i in range(len(atom_specifier)):
            i += 1
            if (
                os.path.isfile(
                    glob.glob(f"{run_loc}/{constr_atoms}{i}/init_2/*restart*")[0]
                )
                is False
            ):
                print(
                    'init_2 restart files not found, please ensure "init_2" has been run'
                )
                raise FileNotFoundError

            if len(control_opts) > 0:
                # Add any additional control options to the hole control file
                parsed_control_opts = fo.get_control_keywords(
                    f"{run_loc}/{constr_atoms}{i}/hole/control.in"
                )
                control_opts = fo.mod_keywords(control_opts, parsed_control_opts)
                control_content = fo.change_control_keywords(
                    f"{run_loc}/{constr_atoms}{i}/hole/control.in", control_opts
                )

                with open(
                    f"{run_loc}/{constr_atoms}{i}/hole/control.in", "w"
                ) as control_file:
                    control_file.writelines(control_content)

            # Copy the restart files to hole from init_2
            os.path.isfile(
                glob.glob(f"{run_loc}/{constr_atoms}{i}/init_2/*restart*")[0]
            )
            os.system(
                f"cp {run_loc}/{constr_atoms}{i}/init_2/*restart* {run_loc}/{constr_atoms}{i}/hole/"
            )

        spec_run_info = ""

    # Run the calculation with a nice progress bar if not already run
    if (
        run_type != "ground"
        and os.path.isfile(f"{run_loc}/{constr_atoms}1/{run_type}/aims.out") == False
        and not hpc
    ):
        # Ensure that aims always runs with the following environment variables:
        os.system("export OMP_NUM_THREADS=1")
        os.system("export MKL_NUM_THREADS=1")

        with click.progressbar(
            range(len(atom_specifier)), label=f"calculating {run_type}:"
        ) as bar:
            for i in bar:
                i += 1
                os.system(
                    f"cd ./{run_loc}/{constr_atoms}{i}/{run_type} && mpirun -n {nprocs} "
                    f"{binary} > aims.out{spec_run_info}"
                )

        print(f"{run_type} calculations completed successfully")

    elif run_type != "ground" and not hpc:
        print(f"{run_type} calculations already completed, skipping calculation...")

    # This needs to be passed to process()
    ctx.obj["RUN_TYPE"] = run_type

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
    "-a", "--atom_index", type=int, help="index of the (first) atom to constrain"
)
@click.option(
    "-o",
    "--occ_type",
    default="deltascf_basis",
    show_default=True,
    type=click.Choice(["deltascf_basis", "force_occupation_basis"]),
    help="select whether the old or new occupation routine is used",
)
@click.option(
    "-u",
    "--multiplicity",
    type=int,
    default=1,
    show_default=True,
    help="set the multiplicity of the system",
)
@click.option(
    "-n",
    "--n_quantum_number",
    "n_qn",
    type=int,
    help="principal quantum number of constrained state",
)
@click.option(
    "-l",
    "--l_quantum_number",
    "l_qn",
    type=int,
    help="orbital momentum quantum number of constrained state",
)
@click.option(
    "-m",
    "--m_quantum_number",
    "m_qn",
    type=int,
    help="magnetic quantum number for projection of orbital momentum",
)
@click.option(
    "-k",
    "--ks_max",
    type=click.IntRange(1),
    help="maximum Kohn-Sham state to constrain",
)
@click.option(
    "-c",
    "--control_opts",
    multiple=True,
    type=str,
    help="provide additional options to be used in 'control.in'",
)
@click.pass_context
def basis(
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
    calc = ctx.obj["CALC"]
    species = ctx.obj["SPECIES"]
    basis_set = ctx.obj["BASIS_SET"]
    nprocs = ctx.obj["NPROCS"]
    binary = ctx.obj["BINARY"]
    hpc = ctx.obj["HPC"]
    constr_atoms = ctx.obj["CONSTR_ATOM"]
    spec_at_constr = ctx.obj["SPEC_AT_CONSTR"]
    occ = ctx.obj["OCC"]

    # Raise a warning if no additional control options have been specified
    if len(control_opts) < 1:
        print(
            "\nWARNING: no control options provided, using default options "
            "which can be found in the 'control.in' file"
        )

    # Convert control options to a dictionary
    control_opts = mu.convert_opts_to_dict(control_opts, None)

    if run_type == "ground":
        # Convert control options to a dictionary
        control_opts = mu.convert_opts_to_dict(control_opts, None)

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

    if (
        run_type == "hole"
        and os.path.isfile(f"{run_loc}/{constr_atoms}1/aims.out") is False
    ):
        mu.check_args(
            ("atom_index", atom_index),
            ("ks_max", ks_max),
            ("n_qn", n_qn),
            ("l_qn", l_qn),
            ("m_qn", m_qn),
        )

        if len(spec_at_constr) == 0 and len(constr_atoms) == 0:
            raise click.BadParameter(
                "ERROR: no atoms have been specified to constrain, please use the "
                "-a/--constr_atoms or -s/--spec_at_constr options"
            )

        # Ensure that aims always runs with the following environment variables:
        os.system("export OMP_NUM_THREADS=1")
        os.system("export MKL_NUM_THREADS=1")

        if os.path.isfile(f"{run_loc}/ground/aims.out") == False:
            raise FileNotFoundError(
                "\nERROR: ground aims.out not found, please ensure the ground calculation has been "
                "run"
            )

        # Create a list of element symbols to constrain
        if len(spec_at_constr) > 0:
            element_symbols = mu.get_element_symbols(ground_geom, spec_at_constr)
        else:
            element_symbols = constr_atoms

        # Makes following code simpler if everything is assumed to be a list
        if type(constr_atoms) is not list:
            list_constr_atoms = list(constr_atoms)
        else:
            list_constr_atoms = constr_atoms

        # Create the directories required for the hole calculation
        fo = ForceOccupation(
            element_symbols,
            run_loc,
            ground_geom,
            control_opts,
        )

        # Get atom indices from the ground state geometry file
        atom_specifier = fo.read_ground_inp(
            list_constr_atoms, spec_at_constr, ground_geom
        )

        basis = Basis(fo)
        basis.setup_basis(multiplicity, n_qn, l_qn, m_qn, occ, ks_max, occ_type)

        # Add molecule identifier to hole geometry.in
        with open(f"{run_loc}{constr_atoms}1/geometry.in", "r") as hole_geom:
            lines = hole_geom.readlines()

        lines.insert(4, f"# {spec_mol}\n")

        with open(f"{run_loc}/{constr_atoms}1/geometry.in", "w") as hole_geom:
            hole_geom.writelines(lines)

        # TODO allow multiple constraints using n_atoms

        for i in range(len(atom_specifier)):
            i += 1

            if len(control_opts) > 0:
                # Add any additional control options to the hole control file
                parsed_control_opts = fo.get_control_keywords(
                    f"{run_loc}{constr_atoms}{i}/control.in"
                )
                control_opts = fo.mod_keywords(control_opts, parsed_control_opts)
                control_content = fo.change_control_keywords(
                    f"{run_loc}{constr_atoms}{i}/control.in", control_opts
                )

                with open(
                    f"{run_loc}{constr_atoms}{i}/control.in", "w"
                ) as control_file:
                    control_file.writelines(control_content)

        if not hpc:
            # Run the hole calculation
            with click.progressbar(
                range(len(atom_specifier)), label="calculating basis hole:"
            ) as prog_bar:
                for i in prog_bar:
                    i += 1
                    os.system(
                        f"cd {run_loc}/{constr_atoms}{i} && mpirun -n "
                        f"{nprocs} {binary} > aims.out"
                    )

    elif os.path.isfile(f"{run_loc}/{constr_atoms}1/aims.out") is True:
        print("hole calculations already completed, skipping calculation...")

    # This needs to be passed to process()
    ctx.obj["RUN_TYPE"] = run_type

    # Compute the dscf energies and plot if option provided
    process(ctx)


if __name__ == "__main__":
    main()
