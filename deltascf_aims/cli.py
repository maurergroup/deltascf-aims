from pathlib import Path
from typing import TYPE_CHECKING, Literal

import click

from deltascf_aims import main
from deltascf_aims.utils.click_extras import (
    MutuallyExclusive,
    MutuallyInclusive,
)

if TYPE_CHECKING:
    from deltascf_aims.core import Start


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
    cls=MutuallyExclusive,
    mutually_exclusive=["geometry_input"],
    type=str,
    help="molecule to be used in the calculation",
)
@click.option(
    "-e",
    "--geometry_input",
    cls=MutuallyExclusive,
    mutually_exclusive=["spec_mol"],
    nargs=1,
    type=click.Path(exists=True, dir_okay=False, writable=True, path_type=Path),
    help="specify a custom geometry.in instead of using a structure from pubchem "
    "or ASE",
)
@click.option(
    "-i",
    "--control_input",
    nargs=1,
    type=click.Path(exists=True, dir_okay=False, writable=True, path_type=Path),
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
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="optionally specify a custom location to run the calculation",
)
@click.option(
    "-c",
    "--constrained_atom",
    "constr_atom",
    type=str,
    # multiple=true,  # TODO: allow for multiple atoms to be constrained
    help="atom to constrain; constrain all atoms of this element",
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
    type=str,
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
    "--run_cmd",
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
def initialise(  # noqa: PLR0913
    ctx: click.Context,
    hpc: bool,
    spec_mol: str,
    geometry_input: Path,
    control_input: Path,
    binary: bool,
    run_location: Path,
    constr_atom: str,
    occupation: float,
    n_atoms: int,
    basis_set: str,
    use_extra_basis: bool,
    print_output: bool,
    force: bool,
    run_cmd: str,
    nprocs: int,
) -> None:
    """
    Automation of core-hole constrained occupation methods in FHI-aims.

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

    Copyright \u00a9 2022-2025, Dylan Morgan dylan.morgan@warwick.ac.uk
    """
    # TODO something like this (but check commented decorator above)
    # if "--help" in sys.argv:
    #     click.echo(ctx.get_help())
    #     raise SystemExit(0)

    return main.start(**locals())


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
    "-s",
    "--spin",
    type=click.Choice([1, 2]),
    default=1,
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
def projector(
    start: "Start",
    run_type: Literal["ground", "init_1", "init_2", "hole"],
    occ_type: Literal["deltascf_projector", "force_occupation_projector"],
    pbc: tuple[int, int, int] | None,
    spin: Literal[1, 2],
    ks_range: tuple[int, int] | None,
    control_opts: tuple[str, ...],
) -> None:
    """Force occupation through defining the Kohn-Sham states to occupy."""
    return main.projector(**locals())


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
    type=click.Choice([1, 2]),
    default=1,
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
    help="provide additional options to be used in 'control.in' in a key=value format",
)
@click.pass_obj
def basis(
    start: "Start",
    run_type: Literal["ground", "hole"],
    occ_type: Literal["deltascf_basis", "force_occupation_basis"],
    spin: Literal[1, 2],
    n_qn: int | None,
    l_qn: int | None,
    m_qn: int | None,
    ks_max: int | None,
    control_opts: tuple[str, ...],
) -> None:
    """Force occupation through basis functions."""
    return main.basis(**locals())


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
def plot(  # noqa: PLR0913
    start: "Start",
    graph: bool,
    intensity: float,
    asym: bool,
    a: float,
    b: float,
    gl_ratio: float,
    omega: float,
    include_name: bool,
    exclude_mabe: bool,
    gmp: float,
) -> None:
    """Plot the simulated XPS spectra."""
    return main.plot(**locals())
