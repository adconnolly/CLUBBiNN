# CLUBBED: subgrid parameterization

CLUBBED (Cloud Layers Unified By Binormals and Equation Discovery) is an equation discovery (ED) extension of Cloud Layers Unified By Binormals (CLUBB).

CLUBB is a boundary-layer cloud parameterization scheme
principally developed by the research group of Vince Larson and subject to the copyright detailed in
[`LICENSE_CLUBB`](https://github.com/adconnolly/CLUBBED/blob/main/LICENSE_CLUBB).
The source code for the latest release can be found at [github/larson-group/clubb_release](https://github.com/larson-group/clubb_release).

This repository contains a Python package (`subgrid_parameterization`) of code used in developing
the CLUBBED extension to CLUBB.
It also contains a number of notebooks used in data analysis and training of the schemes.


## subgrid_parameterization

`subgrid_parameterization` is a Python package containing functions and utilities that are
repeatedly used in the development of CLUBBED and the associated notebooks.
Its source code is contained inside the `src/subgrid_parameterization/` directory.

It can be installed as a Python package using pip by cloning this repository, navigating to it,
and running:
```bash
pip install .
```
It is strongly recommended that this is done from within a [virtual environment](https://docs.python.org/3/library/venv.html).

### Development and optional dependencies

To run the notebooks install `jupyterlab` and other associated dependencies using:
```bash
pip install .[notebooks]
```

For development, create an editable install and include the developer dependencies using:
```bash
pip install -e .[dev]
```
or, if you also want the notebook functionalities:
```bash
pip install -e .[all]
```


## Notebooks

The repository contains several notebooks used for analysis and training in
respective directories.

The analysis notebooks use processed Large-Eddy Simulation (LES) data, e.g. of the BOMEX and DYCOMS cases including turbulence statistics such as Turbulent Kinetic Energy (TKE) as well as the dissipation of TKE, to aid in the design and training of ML schemes.

A coefficient, C14, related to the dissipation of TKE, through the dissipation of horizontal velocity variances, is the initial parameter of interest. Currently, this coefficient is set to a constant value, but high resolution simulations reveal vertical structure and regime dependence. We hope to utilize the same high resolution data to develop a data-driven model of the coefficient. To provide labels for the supervised machine learning of this coefficient, mixing lengths are calculated through python routines which mirror Fortran algorithms used in CLUBB. Other variables are similarly regridded onto a coarse staggered grid, as that used in CLUBB, so that machine learning models ingest only coarse-scale variables. Data-driven models can therefore be implemented in the coarser models with the hopes of improving weather and climate predictions.

The train notebooks aim to provide templates for a general workflow for the construction
of data-driven subgrid parameterizations.
Each one describes the training process for different models for C14 based on different datasets.


## Developer and Contribution Guidelines

Additional developer dependencies can be installed as
[described above](#development-and-optional-dependencies).
These include various tools for enforcing software quality across the project:

- [ruff](https://docs.astral.sh/ruff/) for code formatting and style
- [nb-clean](https://github.com/srstevenson/nb-clean) for removing metadata and empty cells from notebooks

### Testing

Testing is performed using the [`pytest`](https://docs.pytest.org) framework.
All new code to `subgrid_parameterization` should include appropriate unit tests.
Where code handles data integration testing should also be added.

Tests can be found in the `test/` directory and can be run from the top level
using:
```bash
pytest
```

### Continuous Integration

The GitHub repository makes use of continuous integration workflows to check that new
commits conform to style guidelines and pass testing suites.
They are run on pull requests and can be found in the `.github/workflows/` directory.

### Pre-commit

The project uses [`pre-commit`](https://pre-commit.com/) to allow easy
configuration of pre-commit git hooks. They can automatically run formatting
on the files that are staged for commit.
We currently use hooks for applying ruff formatting and cleaning metadata from
notebooks with nb-clean.

After installing the developer dependencies, which include `pre-commit`,
the hooks can be installed with the following command:
```bash
pre-commit install
```

Now the formatting and linting will be applied on each commit.


## Authors and Licensing

The CLUBBED project is led by [Alex Connolly](https://adconnolly.github.io/) of the [Gentine Lab](https://gentinelab.eee.columbia.edu/home)
at Columbia University.

Research Software Engineering support to the project has been provided by Jack Atkinson
and Mikolaj Kowalski of the [Institute of Computing for Climate Science (ICCS)](https://iccs.cam.ac.uk/).

The project is associated with improving the CLUBB code, the license for which can be found in 
[`LICENSE_CLUBB`](https://github.com/adconnolly/CLUBBED/blob/main/LICENSE_CLUBB).
