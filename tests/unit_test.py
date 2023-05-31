#!/usr/bin/env python3

from delta_scf import __version__

import pytest
import os


def test_version():
    assert __version__ == "0.1.0"
