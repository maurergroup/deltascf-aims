#!/usr/bin/env python3

import numpy as np


class DirectionalConstraint:
    """Constrain an atom to move along a given direction only."""

    # Code from ASE https://wiki.fysik.dtu.dk/ase/ase/constraints.html

    def __init__(self, a, direction):
        self.a = a
        self.dir = direction / np.sqrt(np.dot(direction, direction))

    def adjust_positions(self, atoms, new_positions):
        step = new_positions[self.a] - atoms.positions[self.a]
        step = np.dot(step, self.dir)
        new_positions[self.a] = atoms.positions[self.a] + step * self.dir

    def adjust_forces(self, atoms, forces):
        forces[self.a] = self.dir * np.dot(forces[self.a], self.dir)
