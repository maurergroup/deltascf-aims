from typing import Any

from deltascf_aims.core import Basis, Process, Projector, Start


def start(**kwargs: Any) -> None:
    """Entry point for the CLI."""
    # Get context object and arguments for Start
    ctx = kwargs.pop("ctx")
    start = Start(**kwargs)

    start.check_for_geometry_input()
    start.check_for_pbcs()
    start.check_ase_usage()
    start.atoms = start.create_structure()

    curr_path, bin_path = start.check_for_bin()
    bin_path = start.bin_path_prompt(curr_path, bin_path)
    start.check_species_path(bin_path)

    # Return the atoms object with a calculator
    if start.ase:
        start.add_calc(bin_path)

    # TODO pass the Start and Argument objects to the subcommands
    # ctx.obj = {"argument": argument, "start": start}
    ctx.obj = start


def projector(**kwargs: Any) -> None:
    """Automate an FHI-aims core-level constraint calculation run using projector."""
    # Get start object
    start = kwargs["start"]

    # Do this here rather than start to avoid it being called for process which must
    # take constr_atoms not spec_at_constr
    start.check_constr_keywords()

    proj = Projector(**kwargs)

    if (start.found_l_vecs or start.found_k_grid) and proj.pbc is None:
        proj.check_periodic()

    if start.use_extra_basis:
        proj.add_extra_basis_fns(start.constr_atom, start.control_input)

    match proj.run_type:
        case "ground":
            proj.setup_ground(
                start.geometry_input, start.control_input, proj.control_opts, start
            )

            proj.run_ground(
                proj.control_opts,
                start.use_extra_basis,
                start.print_output,
                start.run_cmd,
                start.nprocs,
                start.binary,
                start.atoms.calc,
                start.constr_atom,
            )

        case "init_1":
            atom_specifier, spec_run_info = proj.setup_excited()
            proj.run_excited(atom_specifier, proj.start.constr_atom, "init_1")

        case "init_2":
            atom_specifier, spec_run_info = proj.pre_init_2()
            proj.run_excited(atom_specifier, proj.start.constr_atom, "init_2")

        case "hole":
            atom_specifier, spec_run_info = proj.pre_hole()

            if not start.hpc:  # Don't run on HPC
                proj.run_excited(atom_specifier, proj.start.constr_atom, "hole")


def basis(**kwargs: Any) -> None:
    """Automate an FHI-aims core-level constraint calculation run using basis."""
    # Get start object
    start = kwargs["start"]

    # Do this here rather than start to avoid it being called for process which must
    # take constr_atoms not spec_at_constr
    start.check_constr_keywords()

    basis = Basis(**kwargs)

    if start.use_extra_basis:
        basis.add_extra_basis_fns(start.constr_atom, start.control_input)

    match basis.run_type:
        case "ground":
            basis.setup_ground(
                start.geometry_input, start.control_input, basis.control_opts, start
            )

            basis.run_ground(
                basis.control_opts,
                start.use_extra_basis,
                start.print_output,
                start.run_cmd,
                start.nprocs,
                start.binary,
                start.atoms.calc,
                start.constr_atom,
            )

        case "hole":
            atom_specifier = basis.setup_excited()

            if not start.hpc:  # Don't run on HPC
                basis.run_excited(
                    atom_specifier,
                    basis.start.constr_atom,
                    "hole",
                    basis_constr=True,
                )


def plot(**kwargs: Any) -> None:
    """Plot the XPS spectrum from a DeltaSCF calculation."""
    # Calculate peaks
    process = Process(**kwargs)
    xps, element = process.calc_dscf_energies()

    # Ensure peaks file is in the run location
    if kwargs["start"].run_loc != "./":
        process.move_file_to_run_loc(element, "peaks")

    # Broaden spectrum and write to file
    peaks = process.call_broaden(xps)
    process.write_spectrum_to_file(peaks, element)

    if kwargs["graph"]:
        process.plot_xps(xps)
