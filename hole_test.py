#!/usr/bin/env python3

import os
import socket
import sys
from pathlib import Path

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
    # Search different databases for the specified compound
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


def create_ground_control(procs, binary, species):
    aims_calc = Aims(
        xc="pbe",
        spin="collinear",
        default_initial_moment=0,
        aims_command=f"mpirun -n {procs} {binary}",
        species_dir=f"{species}defaults_2020/tight/",
    )

    return aims_calc


def projector_run_aims(
    run_type,
    atoms,
    pbc,
    aims_calc,
    constr_atom,
    ks_states,
    n_holes,  # Not used for now but will be useful when aims works for multiple core holes
    procs,
    binary,
    species,
    occ_type,
    n_atoms,
    spec_mol,
):

    spec_run_info = None

    if pbc == True:
        atoms.set_pbc(True)

    if run_type == "ground":
        # Create the ground directory if it doesn't already exist
        os.system("mkdir -p test_dirs/ground")

        # Attach the calculator to the atoms object
        atoms.calc = aims_calc

        if os.path.isfile(f"test_dirs/ground/aims.out") == False:
            # Run the ground state calculation
            print("running calculation...")
            atoms.get_potential_energy()
            print("ground calculation completed successfully")

            # Move files to ground directory
            os.system(
                "mv geometry.in control.in aims.out parameters.ase test_dirs/ground/"
            )
        else:
            print("aims.out file found in ground calculation directory")
            print("skipping calculation...")

    if run_type == "init_1":
        basis_set = "tight"
        element_symbols, read_atoms = read_ground_inp(
            constr_atom, "test_dirs/ground/geometry.in"
        )
        at_num, valence = get_electronic_structure(element_symbols, constr_atom)
        nucleus, n_index, valence_index = setup_init_1(
            basis_set,
            species,
            constr_atom,
            read_atoms,
            "./test_dirs/",
            at_num,
            valence,
        )
        setup_init_2(
            ks_states,
            "./test_dirs/",
            constr_atom,
            n_atoms,
            at_num,
            valence,
            n_index,
            valence_index,
            occ_type,
        )
        setup_hole(
            "./test_dirs/",
            ks_states,
            constr_atom,
            n_atoms,
            nucleus,
            valence,
            n_index,
            valence_index,
        )

        spec_run_info = ""

    if run_type == "init_2":
        # Catch for if init_1 hasn't been run
        for i in range(1, n_atoms + 1):
            if (
                os.path.isfile(f"test_dirs/{constr_atom}{i}/init_1/restart_file")
                is False
            ):
                print(
                    'init_1 restart files not found, please ensure "init_1" has been run'
                )
                raise FileNotFoundError

        # Move the restart files to init_1
        for i in range(1, n_atoms + 1):
            os.path.isfile(f"test_dirs/{constr_atom}{i}/init_1/restart_file")
            os.system(
                f"cp test_dirs/{constr_atom}{i}/init_1/restart* test_dirs/{constr_atom}{i}/init_2/"
            )

        # Prevent SCF not converged errors from printing
        spec_run_info = " 2>/dev/null"

    if run_type == "hole":
        # Add molecule identifier to hole geometry.in
        with open(f"test_dirs/{constr_atom}1/hole/geometry.in", "r") as hole_geom:
            lines = hole_geom.readlines()

        lines.insert(4, f"# {spec_mol}\n")

        with open(f"test_dirs/{constr_atom}1/hole/geometry.in", "w") as hole_geom:
            hole_geom.writelines(lines)

        # Catch for if init_2 hasn't been run
        for i in range(1, n_atoms + 1):
            if (
                os.path.isfile(f"test_dirs/{constr_atom}{i}/init_2/restart_file")
                is False
            ):
                print(
                    'init_2 restart files not found, please ensure "init_2" has been run'
                )
                raise FileNotFoundError

        # Move the restart files to hole
        for i in range(1, n_atoms + 1):
            os.path.isfile(f"test_dirs/{constr_atom}{i}/init_2/restart_file")
            os.system(
                f"cp test_dirs/{constr_atom}{i}/init_2/restart* test_dirs/{constr_atom}{i}/hole/"
            )

        spec_run_info = ""

    # Run the calculation with a nice progress bar if not already run
    if (
        run_type != "ground"
        and os.path.isfile(f"test_dirs/{constr_atom}1/{run_type}/aims.out") == False
    ):
        with click.progressbar(
            range(1, n_atoms + 1), label=f"calculating {run_type}:"
        ) as bar:
            for i in bar:
                os.system(
                    f"cd ./test_dirs/{constr_atom}{i}/{run_type} && mpirun -n {procs} {binary} > aims.out{spec_run_info}"
                )

        print(f"{run_type} calculations completed successfully")

    elif run_type != "ground":
        print(f"{run_type} calculations already completed, skipping calculation...")


def basis_run_aims(
    run_type,
    atoms,
    aims_calc,
    constr_atom,
    n_atoms,
    occ_type,
    nprocs,
    binary,
    spec_mol,
):

    atoms.calc = aims_calc

    if run_type == "ground":
        # Create the ground directory if it doesn't already exist
        os.system("mkdir -p test_dirs/ground")

        if os.path.isfile(f"test_dirs/ground/aims.out") == False:
            # Run the ground state calculation
            print("running calculation...")
            atoms.get_potential_energy()
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
        and os.path.isfile(f"test_dirs/{constr_atom}1/hole/aims.out") == False
    ):
        setup_fob(constr_atom, n_atoms, occ_type)

        # Add molecule identifier to hole geometry.in
        with open(f"test_dirs/{constr_atom}1/hole/geometry.in", "r") as hole_geom:
            lines = hole_geom.readlines()

        lines.insert(4, f"# {spec_mol}\n")

        with open(f"test_dirs/{constr_atom}1/hole/geometry.in", "w") as hole_geom:
            hole_geom.writelines(lines)

        # Run the hole calculation
        with click.progressbar(
            range(1, n_atoms + 1), label="calculating basis hole:"
        ) as bar:
            for i in bar:
                os.system(
                    f"cd ./test_dirs/{constr_atom}{i}/hole/ && mpirun -n {nprocs} {binary} > aims.out"
                )


def check_args(run_type, spec_mol, constr_atom, ks_start, ks_stop, n_atoms):
    """Check supplied arguments and produce errors and warnings."""

    # Ground calculation
    if (
        run_type == "ground"
        and spec_mol is None
        and os.path.isfile(f"test_dirs/ground/geometry.in") is False
    ):
        raise click.BadParameter(
            "must specify the molecule for the ground state calculation"
        )

    if run_type == "init_1":
        if constr_atom is None:
            raise click.BadParameter(
                "must specify the atom to constrain for the init_1 calculation"
            )
        if ks_start is None:
            raise click.BadParameter(
                "must specify the Kohn-Sham states to constrain for first initialisation calculation"
            )
        if ks_stop is None:
            raise click.BadParameter(
                "must specify the Kohn-Sham states to constrain for first initialisation calculation"
            )

    # init_2 and hole calculations
    if run_type == "init_2" or run_type == "hole":
        if constr_atom is None:
            raise click.BadParameter(
                f"must specify the atom to constrain for the {run_type} calculation"
            )
        if n_atoms is None:
            raise click.BadParameter(
                f"must specify the number of atoms to constrain for the {run_type} calculation"
            )

    # init_1 calculations
    if run_type != "init_1":
        if ks_start is not None:
            print(
                f"warning: Kohn-Sham states will be ignored for {run_type} calculations"
            )
        if ks_stop is not None:
            print(
                f"warning: Kohn-Sham states will be ignored for {run_type} calculations"
            )

    if run_type == "ground" and constr_atom is not None:
        print(f"warning: constrained atom will be ignored for {run_type} calculations")


@click.command()
@click.option(
    "-m",
    "--molecule",
    "spec_mol",
    type=str,
    required=False,
    help="molecule to be used in the calculation",
)
@click.option(
    "-p", "--pbc", is_flag=True, help="create a cell with periodic boundary conditions"
)
@click.option(
    "-r",
    "--run_type",
    type=click.Choice(["ground", "init_1", "init_2", "hole"]),
    required=True,
    help="select the type of calculation to perform",
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
    "-c", "--constrained_atom", "constr_atom", type=str, help="atom to constrain"
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
@click.option(
    "-t",
    "--occ_type",
    required=True,
    type=click.Choice(["old_projector", "new_projector", "old_basis", "new_basis"]),
)
@click.option("-a", "--n_atoms", type=int, help="the number of atoms to constrain")
@click.option("-g", "--graph", is_flag=True, help="print out the simulated XPS spectra")
@click.option(
    "-d", "--debug", is_flag=True, help="for developer use: print debug information"
)
def main(
    spec_mol,
    pbc,
    run_type,
    nprocs,
    constr_atom,
    ks_start,
    ks_stop,
    occ_type,
    n_atoms,
    graph,
    debug,
):

    # Check the parsed arguments
    check_args(run_type, spec_mol, constr_atom, ks_start, ks_stop, n_atoms)

    # Create a length 2 list of the Kohn-Sham states to constrain
    ks_states = [ks_start, ks_stop]

    # Check if calculation is projector or basis
    projector = False
    basis = False
    if "projector" in occ_type:
        projector = True
    if "basis" in occ_type:
        basis = True

    # Set other parameters
    home_dir = str(Path.home())
    hostname = socket.gethostname()
    species = f"{home_dir}/Programming/mac_projects/FHIaims/species_defaults/"
    binary = None

    # Set different binaries for different machines
    if hostname == "apollo":
        # Change the binary to use if debugging
        if debug:
            binary = f"{home_dir}/Programming/projects/FHIaims/build/aims_debug.x"
        else:
            binary = f"{home_dir}/Programming/projects/FHIaims/build/aims.220915.scalapack.mpi.x"

    elif hostname == "maccie":
        if debug:
            binary = f"{home_dir}/Programming/mac_projects/FHIaims/build/aims_debug.x"
        else:
            binary = (
                f"{home_dir}/Programming/mac_projects/FHIaims/build/aims.220915.mpi.x"
            )

    n_holes = 1

    # Find the structure if not given
    if spec_mol is None:
        atoms = read("./test_dirs/ground/geometry.in")
        print("molecule argument not provided, defaulting to using geometry.in file")
    else:
        atoms = build_geometry(spec_mol)

    aims_calc = create_ground_control(nprocs, binary, species)

    if projector:
        projector_run_aims(
            run_type,
            atoms,
            pbc,
            aims_calc,
            constr_atom,
            ks_states,
            n_holes,
            nprocs,
            binary,
            species,
            occ_type,
            n_atoms,
            spec_mol,
        )

    if basis:
        basis_run_aims(
            run_type,
            atoms,
            aims_calc,
            constr_atom,
            n_atoms,
            occ_type,
            nprocs,
            binary,
            spec_mol,
        )

    # Calculate the delta scf energy and plot
    if run_type == "hole":
        grenrgys = read_ground("test_dirs/")
        element, excienrgys = read_atoms("test_dirs/", constr_atom, contains_number)
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

    if graph:
        print("\nplotting spectrum and calculating MABE...")
        sim_xps_spectrum(constr_atom)


if __name__ == "__main__":
    main()
