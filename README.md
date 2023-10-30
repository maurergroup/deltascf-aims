# deltaSCF-aims

An application to test the development of core level spectroscopy simulation methods in FHI-aims

[![Python Package](https://github.com/maurergroup/deltascf-aims/actions/workflows/python-package.yml/badge.svg)](https://github.com/maurergroup/deltascf-aims/actions/workflows/python-package.yml)
[![Dependency Review](https://github.com/maurergroup/deltascf-aims/actions/workflows/dependency-review.yml/badge.svg)](https://github.com/maurergroup/deltascf-aims/actions/workflows/dependency-review.yml)

## Requirements 

- [python pip](https://pypi.org/project/pip/)
- Access to an FHI-aims binary with the basis sets saved at `FHIaims/species_defaults`. Note that the `FHIaims/` top level directory can be at any writeable location in your filesystem as `deltaSCF-aims` will ask for the location of this.

## Installation

### PyPi

This package is available to download on PyPi. To install, simply execute 

``` shell
pip install deltascf-aims
```

And the binary should be available on your path as `deltascf`.

### Cloning the Repository 

Setup of a virtual environment is automated using poetry. Ensure poetry is installed with:

```shell
pip install poetry
```

It is recommended to use [pyenv](https://github.com/pyenv/pyenv) to manage the local python version, however this is not essential. If this is desired, then simply install and set the local python version. You should also tell poetry the environment you wish to use to create a virtual environment. For example:

```shell
pyenv install 3.11.1
pyenv local 3.11.1
poetry env use 3.11
```

Then install the virtual environment:

```shell
poetry install
```

Then either enter the poetry virtual environment with `poetry shell`, and use deltaSCF-aims, as described below, or prefix commands to deltaSCF-aims with `poetry run`.

## Usage

It is necessary to have a compiled FHI-aims binary. The location needs to be specified and is then saved by the application. If you wish to change the binary name/location, simply invoke the app with the `-b` option.

The click library has been used to parse command line arguments. To view all the options, run `deltascf` with the help flag:

```shell
deltascf --help
```
More extensible documentation of usage will be provided at a later date. 
