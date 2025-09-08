"""
Pytest test configuration file.

Pytest fixtures that may need to be shared across multiple test files should
be defined here. See:

https://docs.pytest.org/en/stable/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files)
"""

import pytest
import os
import xarray as xr
import hashlib
import warnings

from pathlib import Path


def compute_mdf5_checksum(file_path):
    """Compute the MD5 checksum of a file."""

    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


@pytest.fixture(scope="session")
def bomex_dataset():
    """Provides the BOMEX NetCDF file to tests."""
    tests_dir = Path(__file__).parent
    path = tests_dir / "data" / "BOMEX_pruned_thinned.nc"

    if not path.exists():
        raise FileNotFoundError(f"BOMEX dataset not found at '{path}'")

    return xr.open_dataset(path)


@pytest.fixture(scope="session")
def bomex_mixing_length():
    """
    Provides reference `compute_mixing_length` evaluations from Fortran code.

    The data was obtained by running CLUBB standalone calculation and logging
    input/outputs on each invocation of the Fortran function from:

    https://github.com/m2lines/clubb_ML/blob/master/src/CLUBB_core/mixing_length.F90

    Each input/output is stored under a key that matches its Fortran variable
    name. 'samples' dimension refers to individual, logged invocations of the
    function.

    The modified CLUBB with data logging is available here:

    https://github.com/m2lines/clubb_ML/tree/log-mixing-length

    """
    path_prefix = Path(__file__).parent / "data"
    path = path_prefix / "bomex_mixing_length_calculation_samples.nc"

    if not path.exists():
        raise FileNotFoundError(
            f"BOMEX mixing length reference input not found at '{path}'"
        )

    return xr.open_dataset(path)
