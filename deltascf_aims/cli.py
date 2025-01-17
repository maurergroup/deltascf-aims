import click

import deltascf_aims.main as main
from deltascf_aims.utils.click_extras import (
    MutuallyExclusive,
    MutuallyInclusive,
    NotRequiredIf,
)


@click.group()
# @click.argument("argument", cls=showhelpsubcmd)
@click.option(
    "-h",
    "--hpc",
    cls=MutuallyExclusive,
    mutually_exclusive=["binary"],
    is_flag=True,
    help="setup a calculation primarily for use on a hpc cluster without running "
    "the calculation",
)
@click.option(
    "-m",
    "--molecule",
    "spec_mol",
    cls=NotRequiredIf,
    not_required_if=["geometry_input"],
    type=str,
    help="molecule to be used in the calculation",
)
@click.option(
    "-e",
    "--geometry_input",
    cls=NotRequiredIf,
    not_required_if=["spec_mol"],
    nargs=1,
    type=click.File(),
    help="specify a custom geometry.in instead of using a structure from pubchem "
    "or ASE",
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
    cls=NotRequiredIf,
    not_required_if=["spec_at_constr"],
    type=str,
    # multiple=true,  # TODO: allow for multiple atoms to be constrained
    help="atom to constrain; constrain all atoms of this element",
)
@click.option(
    "-s",
    "--specific_atom_constraint",
    "spec_at_constr",
    cls=NotRequiredIf,
    not_required_if=["constr_atom"],
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
@click.option(
    "-p",
    "--print_output",
    is_flag=True,
    help="print the live output of the calculation",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="force the calculation to run even if it has already been run",
)
@click.option(
    "--aims_cmd",
    default="mpirun -n",
    show_default=True,
    nargs=1,
    type=str,
    help="parallel command to run FHI-aims",
)
@click.option(
    "-n",
    "--nprocs",
    default=4,
    nargs=1,
    show_default=True,
    type=int,
    help="number of processors to use",
)
@click.version_option()
@click.pass_context
def initialise(
    ctx,
    hpc,
    spec_mol,
    geometry_input,
    control_input,
    binary,
    run_location,
    constr_atom,
    spec_at_constr,
    occupation,
    n_atoms,
    basis_set,
    use_extra_basis,
    print_output,
    force,
    aims_cmd,
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

    # TODO something like this (but check commented decorator above)
    # if "--help" in sys.argv:
    #     click.echo(ctx.get_help())
    #     raise SystemExit(0)

    return main.start(
        ctx,
        hpc,
        spec_mol,
        geometry_input,
        control_input,
        binary,
        run_location,
        constr_atom,
        spec_at_constr,
        occupation,
        n_atoms,
        basis_set,
        use_extra_basis,
        print_output,
        force,
        aims_cmd,
        nprocs,
    )


@initialise.command()
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
    help="range of Kohn-Sham states to constrain",
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

    return main.projector(
        start, run_type, occ_type, pbc, l_vecs, spin, ks_range, control_opts
    )


@initialise.command()
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
    required=True,
    type=int,
    help="principal quantum number of constrained state",
)
@click.option(
    "-l",
    "--l_quantum_number",
    "l_qn",
    required=True,
    type=int,
    help="orbital momentum quantum number of constrained state",
)
@click.option(
    "-m",
    "--m_quantum_number",
    "m_qn",
    required=True,
    type=int,
    help="magnetic quantum number for projection of orbital momentum",
)
@click.option(
    "-k",
    "--ks_max",
    required=True,
    type=click.IntRange(1),
    help="maximum Kohn-Sham state to constrain",
)
@click.option(
    "-c",
    "--control_opts",
    multiple=True,
    type=str,
    help="provide additional options to be used in 'control.in' in a key=value format",
)
@click.pass_obj
def basis(start, run_type, occ_type, spin, n_qn, l_qn, m_qn, ks_max, control_opts):
    """
    Force occupation through basis functions.
    """

    return main.basis(
        start, run_type, occ_type, spin, n_qn, l_qn, m_qn, ks_max, control_opts
    )


@initialise.command()
@click.option(
    "-g",
    "--graph",
    is_flag=True,
    default=False,
    show_default=True,
    help="graph the simulated XPS spectra",
)
@click.option(
    "-A",
    "--intensity",
    default=1,
    type=float,
    show_default=True,
    help="set as 1 for all subpeaks if they are non-degenerate",
)
@click.option(
    "-s",
    "--asym",
    is_flag=True,
    default=False,
    show_default=True,
    help="simulate the XPS spectrum with asymmetry",
)
@click.option(
    "-a",
    "--asym_param",
    "a",
    cls=MutuallyInclusive,
    mutually_inclusive=["asym"],
    default=0.2,
    type=float,
    show_default=True,
    help="define the asymmetry parameter",
)
@click.option(
    "-b",
    "--asym_trans_param",
    "b",
    cls=MutuallyInclusive,
    mutually_inclusive=["asym"],
    default=0.0,
    type=float,
    show_default=True,
    help="define the asymmetry translation parameter",
)
@click.option(
    "-m",
    "--gl_ratio",
    default=0.5,
    type=click.FloatRange(min=0, max=1, clamp=True),
    show_default=True,
    help="set the mixing parameter for the Gaussian-Lorentzian functions",
)
@click.option(
    "-o",
    "--omega",
    default=0.35,
    type=click.FloatRange(min=0, max=1, clamp=True),
    show_default=True,
    help="full width at half maximum value",
)
@click.option(
    "-i",
    "--include_name",
    is_flag=True,
    default=False,
    show_default=True,
    help="include the molecule name in the plot",
)
@click.option(
    "-e",
    "--exclude_mabe",
    is_flag=True,
    default=False,
    show_default=True,
    help="exclude the mean average binding energy from the plot",
)
@click.option(
    "--gmp",
    default=0.003,
    type=click.FloatRange(min=0, max_open=True),
    show_default=True,
    help="global minimum percentage",
)
@click.pass_obj
def plot(
    start,
    graph,
    intensity,
    asym,
    a,
    b,
    gl_ratio,
    omega,
    include_name,
    exclude_mabe,
    gmp,
):
    """
    Plot the simulated XPS spectra.
    """

    return main.plot(
        start,
        graph,
        intensity,
        asym,
        a,
        b,
        gl_ratio,
        omega,
        include_name,
        exclude_mabe,
        gmp,
    )
