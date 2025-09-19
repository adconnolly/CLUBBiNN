# CLUBBED: subgrid parameterization

CLUBBED (Cloud Layers Unified By Binormals and Equation Discovery) is an equation discovery (ED) extension of Cloud Layers Unified By Binormals (CLUBB).

CLUBB is a boundary-layer cloud parameterization scheme
principally developed by the research group of Vince Larson and subject to the copyright detailed in
[LICENSE_CLUBB](https://github.com/adconnolly/CLUBBED/blob/main/LICENSE_CLUBB).
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






# Pre-commit hooks
The project uses [`pre-commit`](https://pre-commit.com/) to allow easy
configuration of pre-commit git hooks. They can automatically run formatting
on the files that are staged for commit. To install the pre-commit hooks,
first install the package:

```bash
pip install pre-commit
```

Then, run the following command to install the hooks:

```bash
pre-commit install
```

Now the formatting and linting will be applied on each commit.
Specifically we use:

- [ruff](https://docs.astral.sh/ruff/) to apply standardized formatting to any Python code
- [nb-clean](https://github.com/srstevenson/nb-clean) to clean notebooks whilst preserving the outputs
