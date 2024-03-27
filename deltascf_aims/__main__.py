import deltascf_aims.cli as cli
from deltascf_aims.aims_dscf import BasisWrapper, Process, ProjectorWrapper, Start


@cli.Initialise
def start(*args):

    # TODO
    # print(sys.argv)

    # if "--help" in sys.argv:
    #     tmp = cli.get_help(ctx)

    #     print(tmp)

    # Get context object and arguments for Start
    ctx = list(args).pop(0)
    start = Start(*args)

    # start.check_for_help_arg()
    if len(start.spec_at_constr) > 0:
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


@cli.Projector
def projector(*args):

    # Do this here rather than start to avoid it being called for process which must
    # take constr_atoms not spec_at_constr
    start.check_constr_keywords()

    proj = ProjectorWrapper(
        # start, run_type, occ_type, pbc, l_vecs, spin, ks_range, control_opts
        *args
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
            raise ValueError(f"Invalid run_type: {proj.type_run}")


@cli.Basis
def basis(*args):

    # Do this here rather than start to avoid it being called for process which must
    # take constr_atoms not spec_at_constr
    start.check_constr_keywords()

    basis = BasisWrapper(
        # start,
        # run_type,
        # occ_type,
        # spin,
        # n_qn,
        # l_qn,
        # m_qn,
        # ks_max,
        # control_opts,
        *args
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
            atom_specifier = basis.setup_excited()

            if not start.hpc:  # Don't run on HPC
                basis.run_excited(
                    atom_specifier, basis.constr_atoms, "hole", None, basis_constr=True
                )

        case _:
            raise ValueError(f"Invalid run_type: {basis.run_type}")


@cli.Plot
def plot(*args):

    # Calculate peaks
    process = Process(*args)
    xps, element = process.calc_dscf_energies()

    # Ensure peaks file is in the run location
    if start.run_loc != "./":
        process.move_file_to_run_loc(element, "peaks")

    # Broaden spectrum and write to file
    peaks = process.call_broaden(xps)
    process.write_spectrum_to_file(peaks, element)

    graph = args[0]
    if graph:
        process.plot_xps(xps)
