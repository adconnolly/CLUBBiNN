# CLUBBED
Cloud Layers Unified By Binormals and Equation Discovery (CLUBBED) is an equation discovery (ED) extension of Cloud Layers Unified By Binormals (CLUBB).

CLUBB (github.com/larson-group/clubb_release) is principally developed by the research group of Vince Larson and subject to the following copyright.

```
********************************************************************************
*                           Copyright Notice
*                       This code is (C) 2006-2024
*                 Vincent E. Larson and Brian M. Griffin
*
*         The distribution of this code and derived works thereof 
*                      should include this notice.
*
*         Portions of this code derived from other sources (Hugh Morrison,
*         ACM TOMS, Numerical Recipes, et cetera) are the intellectual
*         property of their respective authors as noted and are also 
*         subject to copyright. 
********************************************************************************
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
