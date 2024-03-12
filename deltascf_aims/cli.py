#!/usr/bin/env python3

import click

from deltascf_aims.aims_dscf import BasisWrapper, Process, ProjectorWrapper, Start
from dscf_utils.custom_click import MutuallyExclusive, MutuallyInclusive


@click.group()
# @click.argument("argument", cls=ShowHelpSubCmd)
@click.option(
    "-h",
    "--hpc",
    cls=MutuallyExclusive,
    mutually_exclusive=["--binary"],
    is_flag=True,
    help="setup a calculation primarily for use on a HPC cluster WITHOUT "
    "running the calculation",
)
@click.option(
    "-m",
    "--molecule",
    "spec_mol",
    cls=MutuallyExclusive,
    mutually_exclusive=["--geometry_input"],
    type=str,
    help="molecule to be used in the calculation",
)
@click.option(
    "-e",
    "--geometry_input",
    cls=MutuallyExclusive,
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
    cls=MutuallyExclusive,
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
    help="optionally specify a custom location to run the calculation",
)
@click.option(
    "-c",
    "--constrained_atom",
    "constr_atom",
    cls=MutuallyExclusive,
    mutually_exclusive=["spec_at_constr"],
    type=str,
    # multiple=True,  # TODO: allow for multiple atoms to be constrained
    help="atom to constrain; constrain all atoms of this element",
)
@click.option(
    "-s",
    "--specific_atom_constraint",
    "spec_at_constr",
    cls=MutuallyExclusive,
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
@click.option(
    "-x",
    "--use_extra_basis",
    is_flag=True,
    help="add additional basis functions for the core hole",
)
# @click.option("-g", "--graph", is_flag=True, help="print out the simulated XPS spectra")
@click.option(
    "-p",
    "--print_output",
    is_flag=True,
    help="print the live output of the calculation",
    # TODO
)
@click.option(
    "-n",
    "--nprocs",
    default=4,
    show_default=True,
    type=int,
    help="number of processors to use",
)
@click.version_option()
@click.pass_context
def cli(
    ctx,
    # argument,
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
    use_extra_basis,
    # graph,
    print_output,
    nprocs,
):
    """
    An interface to automate core-hole constrained occupation methods in
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

    Copyright \u00A9 2022-2024, Dylan Morgan dylan.morgan@warwick.ac.uk
    """

    # TODO
    # print(sys.argv)

    # if "--help" in sys.argv:
    #     tmp = cli.get_help(ctx)

    #     print(tmp)

    start = Start(
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
        use_extra_basis,
        # graph,
        print_output,
        nprocs,
    )

    # start.check_for_help_arg()
    if len(spec_at_constr) > 0:
        start.check_for_geometry_input()

    start.check_for_pbcs()
    start.check_ase_usage()
    start.atoms = start.create_structure()

    if start.constr_atom is None:
        start.find_constr_atom_element(start.atoms)

    curr_path, bin_path = start.check_for_bin()
    bin_path = start.bin_path_prompt(curr_path, bin_path)
    start.check_species_path(bin_path)

    # Return the atoms object with a calculator
    if start.ase:
        start.atoms = start.add_calc(start.atoms, bin_path)

    # pass the Start and Argument objects to the subcommands
    # ctx.obj = {"argument": argument, "start": start}
    ctx.obj = start


@cli.command()
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
    nargs=3,
    help="provide the lattice vectors as 3 vectors of length 3",
)
@click.option(
    "-s",
    "--spin",
    type=click.Choice(["1", "2"]),
    default="1",
    show_default=True,
    help="set the spin channel of the constraint",
)
@click.option(
    "-k",
    "--ks_range",
    nargs=2,
    type=click.IntRange(1),
    help="range of Kohn-Sham states to constrain - taken with 2 arguments",
)
@click.option(
    "-c",
    "--control_opts",
    multiple=True,
    type=str,
    help="provide additional options to be used in 'control.in' in a key=value format",
)
@click.pass_obj
def projector(start, run_type, occ_type, pbc, l_vecs, spin, ks_range, control_opts):
    """
    Force occupation through defining the Kohn-Sham states to occupy.
    """

    proj = ProjectorWrapper(
        start, run_type, occ_type, pbc, l_vecs, spin, ks_range, control_opts
    )

    if start.found_l_vecs or start.found_k_grid:
        if proj.pbc is None:
            proj.check_periodic()

    # If not ground, all the geometry.in files have been written already
    if proj.l_vecs is not None and proj.run_type != "ground":
        proj.add_l_vecs(start.geometry_input)

    if start.use_extra_basis:
        proj.add_extra_basis_fns(start.constr_atom)

    match proj.run_type:
        case "ground":
            proj.setup_ground(start.geometry_input, start.control_input)

            # If ground, geometry.in files haven't been written until after setup_ground
            if proj.l_vecs is not None and not start.ase:
                proj.add_l_vecs(start.geometry_input)

            proj.run_ground(
                proj.control_opts,
                start.use_extra_basis,
                proj.l_vecs,
                start.print_output,
                start.nprocs,
                start.binary,
                start.atoms.calc,
            )

        case "init_1":
            atom_specifier, spec_run_info = proj.setup_excited()
            proj.run_excited(atom_specifier, proj.constr_atoms, "init_1", spec_run_info)

        case "init_2":
            atom_specifier, spec_run_info = proj.pre_init_2()
            proj.run_excited(atom_specifier, proj.constr_atoms, "init_2", spec_run_info)

        case "hole":
            atom_specifier, spec_run_info = proj.pre_hole()

            if not start.hpc:  # Don't run on HPC
                proj.run_excited(
                    atom_specifier, proj.constr_atoms, "hole", spec_run_info
                )

        case _:
            raise ValueError(f"Invalid run_type: {proj.run_type}")


@cli.command()
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
    "-s",
    "--spin",
    type=click.Choice(["1", "2"]),
    default="1",
    show_default=True,
    help="set the spin channel for the constraint",
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
@click.pass_obj
def basis(
    start,
    run_type,
    atom_index,
    occ_type,
    spin,
    n_qn,
    l_qn,
    m_qn,
    ks_max,
    control_opts,
):
    """
    Force occupation of Kohn-Sham states through basis functions.
    """

    basis = BasisWrapper(
        start,
        run_type,
        atom_index,
        occ_type,
        spin,
        n_qn,
        l_qn,
        m_qn,
        ks_max,
        control_opts,
    )

    if start.use_extra_basis:
        basis.add_extra_basis_fns(start.constr_atom)

    match basis.run_type:
        case "ground":
            basis.setup_ground(start.geometry_input, start.control_input)

            basis.run_ground(
                basis.control_opts,
                start.use_extra_basis,
                None,
                start.print_output,
                start.nprocs,
                start.binary,
                start.atoms.calc,
            )

        case "hole":
            atom_specifier, _ = basis.setup_excited()

            if not start.hpc:  # Don't run on HPC
                basis.run_excited(atom_specifier, basis.constr_atoms, "hole", None)

        case _:
            raise ValueError(f"Invalid run_type: {basis.run_type}")


@cli.command()
@click.option(
    "-A",
    "--intensity",
    default=1,
    type=float,
    show_default=True,
    help="Set as 1 for all subpeaks if they are non-degenerate",
)
@click.option(
    "-s",
    "--asym",
    is_flag=True,
    default=False,
    show_default=True,
    help="Simulate the XPS spectrum with asymmetry",
)
@click.option(
    "-a",
    "--asym_param",
    "a",
    cls=MutuallyInclusive,
    mutually_inclusive=["--asym"],
    default=0.2,
    type=float,
    show_default=True,
    help="Define the asymmetry parameter",
)
@click.option(
    "-b",
    "--asym_trans_param",
    "b",
    cls=MutuallyInclusive,
    mutually_inclusive=["--asym"],
    default=0.0,
    type=float,
    show_default=True,
    help="Define the asymmetry translation parameter",
)
@click.option(
    "-m",
    "--gl_ratio",
    default=0.5,
    type=click.FloatRange(min=0, max=1, clamp=True),
    show_default=True,
    help="Set the mixing parameter for the Gaussian-Lorentzian functions",
)
@click.option(
    "-o",
    "--omega",
    default=0.35,
    type=click.FloatRange(min=0, max=1, clamp=True),
    show_default=True,
    help="Full width at half maximum value",
)
@click.option(
    "-g",
    "--gmp",
    default=0.003,
    type=click.FloatRange(min=0, max_open=True),
    show_default=True,
    help="Global minimum percentage",
)
@click.pass_obj
def plot(start, intensity, asym, a, b, gl_ratio, omega, gmp):
    """
    Plot the simulated XPS spectra.
    """

    # Calculate peaks
    process = Process(start, intensity, asym, a, b, gl_ratio, omega, gmp)
    xps, element = process.calc_dscf_energies()

    # Ensure peaks file is in the run location
    if start.run_location != "./":
        process.move_file_to_run_loc(element, "peaks")

    # Broaden spectrum and write to file
    peaks = process.call_broaden(xps)
    process.write_spectrum_to_file(peaks, element)

    if start.graph:
        process.plot_xps(xps)
