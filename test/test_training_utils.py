import pytest

import xarray as xr


import subgrid_parameterization.util.training as train


@pytest.fixture
def netcdf_files(tmp_path):
    """Create a temporary NetCDF file to read during a test.

    Provides two paths:
        - `path_with_suffix` that ends with '.nc'
        - `path_no_suffix` that does not have netcdf file extension
    """
    path_with_suffix = tmp_path / "temp.nc"
    path_no_suffix = tmp_path / "temp_netcdf"

    # We don't really care about the contents
    # We fill with some reasonable data
    ds = xr.Dataset(
        data_vars={
            "var": (["x"], [3] * 10),
        },
        coords={"x": [i for i in range(10)]},
    )
    ds.to_netcdf(path_with_suffix)
    ds.to_netcdf(path_no_suffix)
    return path_with_suffix, path_no_suffix


def test_read_as_xarray(netcdf_files):
    path_suffix, path_no_suffix = netcdf_files

    with pytest.raises(FileNotFoundError):
        train.read_as_xarray("Random/Path")
    with pytest.raises(ValueError):
        train.read_as_xarray(path_suffix.parent)
    with pytest.warns(UserWarning):
        train.read_as_xarray(path_no_suffix)

    assert isinstance(train.read_as_xarray(path_suffix), xr.Dataset)
