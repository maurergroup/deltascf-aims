#!/usr/bin/env python3

import os
import warnings
from pathlib import Path

import pytest


def binary_check():
    with open("../deltascf-aims/aims_bin_loc.txt", "r") as binary:
        try:
            bin_path = binary.readlines()[0][:-1]
        except IndexError:
            bin_path = ""

    if Path(bin_path).is_file() == True:
        bin_exists = True
    else:
        warnings.warn("Binary not found in aims_bin_loc.txt")
        bin_exists = False

    assert bin_exists == True


def water_ground():
    os.system("deltascf -m H2O basis -r ground")

    # TODO
    # assert
