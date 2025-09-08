"""
Tests that match the Python implementation of `compute_mixing_length` vs. Fortran.

Refer to `bomex_mixing_length` fixture for details on how the reference data was
collected.
"""

from subgrid_parameterization.preprocess.mixing_length import compute_mixing_length

import xarray as xr
import numpy as np
from typing import Any


def map_mixing_length_argument_types(v: Any) -> Any:
    """
    Convert arguments to ones expected by `compute_mixing_length`.

    `compute_mixing_length` does not take xarray.DataArray at the moment
    due to `newaxis` indexing (via None) which is not supported byXarray.
    """
    dispatch = {
        xr.DataArray: lambda x: x.to_numpy(),
        int: lambda x: x,
        np.int32: lambda x: x,
        np.ndarray: lambda x: x,
    }

    def conversion_error(v):
        raise ValueError(
            f"XArray dataset Numpy inputs conversion not defined for {type(v)}"
        )
        return np.zeros([1])  # Pretend it can return to type checker?

    return dispatch.get(type(v), conversion_error)(v)


def sample_dataset_to_arguments(sample_data: xr.Dataset) -> dict[str, Any]:
    """
    Translate single sample data to argument list.

    We do it manually since we expect the argument names on Python side to change
    and we need to provide a translation. Also we probably don't want to provide
    to invocation more arguments that it accepts!

    Also we convert the arguments to their expected types.
    """
    kwargs = {
        "nzm": sample_data.attrs["nzm"],
        "nzt": sample_data.attrs["nzt"],
        "ngrdcol": 1,  # Data was sampled in one column problem
        # Since zt and zm are dimensions in the dataset they were not
        # expanded by `expand_dims`. We patch that here!
        "zm": sample_data["zm"].expand_dims(dim="column", axis=0),
        "zt": sample_data["zt"].expand_dims(dim="column", axis=0),
        "dzm": sample_data["dzm"],
        "dzt": sample_data["dzt"],
        "invrs_dzm": sample_data["invrs_dzm"],
        "invrs_dzt": sample_data["invrs_dzt"],
        "thvm": sample_data["thvm"],
        "thlm": sample_data["thlm"],
        "rtm": sample_data["rtm"],
        "em": sample_data["em"],
        "Lscale_max": sample_data["Lscale_max"],
        "p_in_Pa": sample_data["p_in_Pa"],
        "exner": sample_data["exner"],
        "thv_ds": sample_data["thv_ds"],
        "mu": sample_data["mu"],
        "lmin": sample_data["lmin"],
        "saturation_formula": sample_data["saturation_formula"],
        "l_implemented": sample_data["l_implemented"],
    }
    return {k: map_mixing_length_argument_types(v) for k, v in kwargs.items()}


def check_single_case(sample_data):
    """Run comparison for a single `compute_mixing_length` Fortran sample."""
    # Pull all attributes and
    kwargs = sample_dataset_to_arguments(sample_data)
    Lscale, Lscale_up, Lscale_down = compute_mixing_length(**kwargs)

    # The maximum error in the test set is about 800 ULPs (~ 1.e-13 relative)
    # but we allow for extra margin (e.g. for future samples)
    tol = 1e-10
    np.testing.assert_allclose(Lscale, sample_data["Lscale"], rtol=tol)
    np.testing.assert_allclose(Lscale_up, sample_data["Lscale_up"], rtol=tol)
    np.testing.assert_allclose(Lscale_down, sample_data["Lscale_down"], rtol=tol)


def test_mixing_length(bomex_mixing_length):
    """
    Test all `compute_mixing_length` samples.

    We need to remove degenerate spatial dimensions (x & y) and create
    dummy `column` dimension to match the Python interface.
    """
    for _, sample_data in bomex_mixing_length.groupby("samples"):
        sample_data = sample_data.squeeze().expand_dims(dim="column", axis=0)
        check_single_case(sample_data)
