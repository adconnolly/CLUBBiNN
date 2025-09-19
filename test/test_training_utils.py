import pytest

import xarray as xr
import numpy as np

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


@pytest.fixture
def reference_z_grids(test_files_dir):
    """Reference z-grids computed from BOMEX dataset for regression check.

    Obtained using the previous version of the pre-processing code.

    """
    z_sam = np.genfromtxt(test_files_dir / "reference_SAM_z_grid.csv", delimiter=",")
    zm = np.genfromtxt(test_files_dir / "reference_CLUBB_zm_grid.csv", delimiter=",")
    zt = np.genfromtxt(test_files_dir / "reference_CLUBB_zt_grid.csv", delimiter=",")
    return z_sam, zm, zt


DATA_CASTERS = {
    "numpy": lambda g: g,  # numpy array
    "xarray": lambda g: xr.DataArray(
        g, dims=["i"], coords={"i": list(range(len(g)))}
    ),  # xarray DataArray
    "list": lambda g: list(g),  # Python list
}


@pytest.mark.parametrize(
    "input_data_type",
    [
        "numpy",
        "xarray",
        "list",
    ],
)
def test_SAMDataInterface_grid(reference_z_grids, input_data_type):
    # It is more of a regression test, we match the results for BOMEX dataset
    # with the previous pre-processing code.
    z_sam, zm_ref, zt_ref = reference_z_grids

    z_arg = DATA_CASTERS[input_data_type](z_sam)
    zm, zt = train.SAMDataInterface.convert_sam_grid_to_clubb(z_arg)
    np.testing.assert_equal(zm, zm_ref)
    np.testing.assert_equal(zt, zt_ref)


def test_SAMDataInterface_grid_errors():
    with pytest.raises(NotImplementedError):
        # 2D grid not supported yet
        train.SAMDataInterface.convert_sam_grid_to_clubb(np.array([[0, 1], [2, 3]]))


def test_SAMDataInterface_init(bomex_dataset):
    # Just a smoke test
    # TODO: Figure out something more sensible
    sam_data = train.SAMDataInterface(bomex_dataset)
    sam_data.zm
    sam_data.zt


def test_SAMDataInterface_init_errors(bomex_dataset):
    ds = bomex_dataset

    # Missing coordinates
    coords_to_remove = ["y", "x", "z", "time"]
    for coord in coords_to_remove:
        with pytest.raises(ValueError):
            train.SAMDataInterface(ds.drop_vars(coord))

    # Non-degenerate y or x dimensions
    for dim in ["y", "x"]:
        # We expand the dataset along a dimension
        temp = ds.copy(deep=True)
        temp.coords[dim].values[0] = ds.coords[dim].values[0] + 1.0
        combined = xr.concat([ds, temp], dim=dim, data_vars="minimal")

        with pytest.raises(ValueError):
            train.SAMDataInterface(combined)
