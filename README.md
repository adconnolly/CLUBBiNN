# CLUBBED: subgrid parameterization

CLUBBED (Cloud Layers Unified By Binormals and Equation Discovery) is an equation discovery (ED) extension of Cloud Layers Unified By Binormals (CLUBB).

CLUBB is a boundary-layer cloud parameterization scheme
principally developed by the research group of Vince Larson and subject to the copyright detailed in
[LICENSE_CLUBB](https://github.com/adconnolly/CLUBBED/blob/main/LICENSE_CLUBB).
The source code for the latest release can be found at [github/larson-group/clubb_release](https://github.com/larson-group/clubb_release).

This repository contains a Python package (`subgrid_parameterization`) of code used in developing
the CLUBBED extension to CLUBB.
It also contains a number of notebooks used in data analysis and training of the schemes.
```
```

# subgrid_parameterization
Template for a general workflow for the construction of ML subgrid parameterizations.

Install using:

`pip install .`

For development (preferred option for now), create an editable install using:

`pip install -e .` 

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
