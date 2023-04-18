# deltaSCF-aims

An application to test the development of core level spectroscopy simulation methods in FHI-aims

[![Python Package](https://github.com/maurergroup/deltascf-aims/actions/workflows/python-package.yml/badge.svg)](https://github.com/maurergroup/deltascf-aims/actions/workflows/python-package.yml)
[![Dependency Review](https://github.com/maurergroup/deltascf-aims/actions/workflows/dependency-review.yml/badge.svg)](https://github.com/maurergroup/deltascf-aims/actions/workflows/dependency-review.yml)

## Installation

Setup of a virtual environment is automated using poetry. Ensure poetry is installed with:

```shell
pip install poetry
```

It is recommended to use [pyenv](https://github.com/pyenv/pyenv) to manage the local python version, however this is not essential. If this is desired, then simply install and set the local python version. You should also tell poetry the environment you wish to use to create a virtual environment:

```shell
pyenv install 3.11.1
pyenv local 3.11.1
poetry env use 3.11
```

Then install the virtual environment:

```shell
poetry install
```

It is also necessary to have a compiled FHI-aims binary. The location needs to be specified and is then saved by the application. If you wish to change the binary name/location, simply invoke the app with the `-b` option.

## Usage

The click library has been used to parse command line arguments. To view all the options, firstly enter the poetry shell and run deltascf with the help flag:

```shell
poetry shell
deltascf --help
```

If you do not wish to enter the shell, deltascf can also be run in a single command

`poetry run deltascf --help`
