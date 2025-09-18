"""Utility functions for training models for C14 prediction."""

import torch
import xarray as xr
import warnings
from pathlib import Path


def read_as_xarray(path: Path | str) -> xr.Dataset:
    """Read a netCDF datafile as an xarray Dataset.

    Parameters
    ----------
    path : Path | str
        Path to the netCDF datafile.

    Returns
    -------
    xarray.Dataset
        Opened NetCDF file.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path '{path}' does not point to a file.")
    if path.suffix != ".nc":
        # TODO, Review: This warning may be useless, since in most cases it will be
        #  immediately followed by some error.
        warnings.warn(
            f"File does not have '.nc' suffix, but {path.suffix}. Are you sure it is a NetCDF file?"
        )

    return xr.open_dataset(path)
