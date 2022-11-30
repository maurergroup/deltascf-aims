# aims-hole-test
An application to test the development of core level spectroscopy simulation methods in FHI-aims

## Installation
Setup of a virtual environment is automated using poetry. Ensure poetry is installed with:

```shell
pip install poetry
```

Then install the virtual environment:

```shell
poetry install --no-root
```

It is also necessary to have a compiled FHI-aims binary and to specify the location of this in `hole_test.py` for now.

## Usage
The click library has been used to parse command line arguments. To view all the options, simply type 

```shell
./hole_test.py --help
```

into the command line
