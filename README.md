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

# Cloning the repository
This project is using [git Large File Storage (LFS)](https://git-lfs.com/)
to manage large dataset files. To clone the repository, you need to have git LFS
installed. Otherwise when you clone the large files in `datasets` folder will 
be replaced with small 'pointer' files like this:

```
version https://git-lfs.github.com/spec/v1
oid sha256:...
size 123456
```

To install git LFS on your system download follow the [instructions](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage). They work only if you have administrative
rights on your system. If you not it is possible to install git LFS for a single
user, but [extra steps](https://gist.github.com/pourmand1376/bc48a407f781d6decae316a5cfa7d8ab)
are required.

After that ordinary `git clone` should fetch the large files in a lazy fashion
(i.e. files will be fetched when they appear in the working tree, e.g. when
checking out a branch that contains them).


# subgrid_parameterization
Template for a general workflow for the construction of ML subgrid parameterizations.

Install using:

`pip install .`

For development (preferred option for now), create an editable install using:

`pip install -e .` 

# Pre-commit hooks
The project uses [`pre-commit`](https://pre-commit.com/) to allow easy 
configuration of pre-commit git hooks. They can automatically run the linting 
and formatting on the files that are staged for commit. To install the
 pre-commit hooks, first install the package:

```bash
pip install pre-commit
```
Or if you are installing the package in editable mode for development
the `pre-commit` will be included in the `dev` dependencies.

Then, run the following command to install the hooks:

```bash
pre-commit install
```

Now the formatting and linting will be applied on each commit. 
