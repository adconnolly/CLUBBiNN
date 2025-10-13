"""Pytest test configuration file.

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
    """Provides the BOMEX NetCdf file to tests.

    BOMEX is loaded from a data directory specified by an environment variable
    `CLUBBED_DATA_DIR`.
    """
    path_prefix = Path(os.getenv("CLUBBED_DATA_PATH"))
    path = path_prefix / "sam-bomex" / "BOMEX_64x64x75_100m_40m_1s.nc"

    if not path.exists():
        raise FileNotFoundError(f"BOMEX dataset not found at '{path}'")

    MD5_SUM = "c2e2f97cfd3fa4c15dcb2a27218960d4"
    if (md5sum := compute_mdf5_checksum(path)) != MD5_SUM:
        warnings.warn(
            f"Checksum of the BOMEX dataset file at '{path}' does not match reference.\n"
            f"Reference md5 checksum:\t{MD5_SUM}\n"
            f"Computed md5 checksum:\t{md5sum}\n"
            "This indicates that the file fetched for the tests was modified with respect "
            "to the original file used to write the tests. As the result the tests may fail."
        )

    return xr.open_dataset(path)
