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


@np.vectorize
def _ramp_function(x):
    """Simple piecewise-linear function with several slopes."""
    if x < 0:
        return 0.0
    elif x < 1.0:
        return x
    return 1.0 + (x - 1.0) * 0.5


def test_SAMDataInterface_interpolation():
    # NOTE: Must contain knots of the _ramp_function
    x_coarse = np.array([-1.0, 0.0, 1.0, 2.0])
    y_coarse = _ramp_function(x_coarse)

    x_fine = np.linspace(-1.0, 3.0, 40)
    y_fine = train.SAMDataInterface.interpolate_with_extrapolation(
        x_fine, x_coarse, y_coarse
    )
    y_ref = _ramp_function(x_fine)

    # We follow GTest conventions for 'narrow' FP tolerance
    np.testing.assert_array_max_ulp(y_ref, y_fine, maxulp=4)


def test_SAMDataInterface_interpolation_errors():
    # Rejected mixed precision call
    input = (
        np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64),
        np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64),
        np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float64),
    )
    for i in range(len(input)):
        # Apply conversion to 32-bit float to each argument in turn
        new_inputs = [
            np.asarray(x, dtype=np.float32) if i == j else x
            for j, x in enumerate(input)
        ]
        with pytest.raises(ValueError):
            train.SAMDataInterface.interpolate_with_extrapolation(*new_inputs)


def test_SAMDataInterface_variable_access(bomex_dataset):
    sam_data = train.SAMDataInterface(bomex_dataset)

    # We don't test the accuracy of the interpolation
    # only errors and some generic properties of the shape of outputs

    # Valid use cases
    res = sam_data.get_sam_variable_on_clubb_grid("U", grid_type="zm")
    assert res.shape == (len(bomex_dataset["time"]), len(sam_data.zm))

    res = sam_data.get_sam_variable_on_clubb_grid("THLM", grid_type="zt")
    assert res.shape == (len(bomex_dataset["time"]), len(sam_data.zt))

    pressure = sam_data.get_sam_pressure_on_clubb_grid(grid_type="zm")
    assert pressure.shape == (len(sam_data.zm),)

    pressure = sam_data.get_sam_pressure_on_clubb_grid(grid_type="zm")
    assert pressure.shape == (len(sam_data.zm),)

    # Invalid use cases
    # Invalid grid type
    with pytest.raises(ValueError):
        sam_data.get_sam_variable_on_clubb_grid("U", grid_type="invalid_grid")
    with pytest.raises(ValueError):
        sam_data.get_sam_pressure_on_clubb_grid(grid_type="invalid_grid")

    # Invalid variable name
    with pytest.raises(ValueError):
        sam_data.get_sam_variable_on_clubb_grid("I am not in dataset", grid_type="zm")

    # Invalid variable dimensions
    with pytest.raises(ValueError):
        sam_data.get_sam_variable_on_clubb_grid("p", grid_type="zm")


class TestSAMDataInterface_interpolation_matrix:
    """Tests related to the SAMDataInterface.create_interpolation_matrix static method."""

    @pytest.mark.parametrize(
        "from_edges, to_edges",
        [
            # Coarse to fine grid refinement
            (
                np.array([0.0, 1.8, 4.2, 6.0]),
                np.array([0.0, 1.1, 2.1, 3.3, 4.1, 5.0, 6.0]),
            ),
            # Fine to coarse grid coarsening
            (
                np.array([0.0, 0.7, 1.0, 1.1, 2.0, 2.3, 3.0]),
                np.array([0.0, 0.8, 2.1, 3.0]),
            ),
        ],
    )
    def test_interpolation_conservation(self, from_edges, to_edges):
        """Test that interpolation conserves the total integral of values."""
        # TODO: Discuss alternative, this numbers may change with numpy versions
        # (Numpy does not guarantee reproducibility I think)
        prng = np.random.default_rng(42)
        from_values = prng.random(len(from_edges) - 1)

        matrix = train.SAMDataInterface.create_interpolation_matrix(
            to_edges, from_edges
        )
        to_values = matrix @ from_values

        from_integral = np.sum(from_values * np.diff(from_edges))
        to_integral = np.sum(to_values * np.diff(to_edges))

        np.testing.assert_array_almost_equal_nulp(from_integral, to_integral, nulp=4)

    def test_interpolation_matrix_to_single_interval(self):
        # Include a degenerate cell in source grid
        from_edges = np.array([0.0, 0.5, 2.0, 2.0, 3.0])
        to_edges = np.array([0.0, 3.0])

        # We compare against matrix so we need to match the rank
        ref_values = np.expand_dims(
            np.diff(from_edges) / (to_edges[-1] - to_edges[0]), axis=0
        )

        matrix = train.SAMDataInterface.create_interpolation_matrix(
            to_edges, from_edges
        )
        np.testing.assert_array_max_ulp(ref_values, matrix, maxulp=4)

    def test_interpolation_matrix_to_degenerate_interval(self):
        from_edges = np.array([0.0, 1.0, 2.0, 3.0])
        to_edges = np.array([1.5, 1.5])

        ref_values = np.array([[0.0, 0.0, 0.0]])

        matrix = train.SAMDataInterface.create_interpolation_matrix(
            to_edges, from_edges
        )
        np.testing.assert_array_max_ulp(ref_values, matrix, maxulp=4)

    def test_interpolation_matrix_staggered_grids(self):
        from_edges = np.array([0.0, 1.0, 2.0, 3.0])
        to_edges = np.array([0.5, 1.5, 2.5])

        ref_values = np.array(
            [
                [0.5, 0.5, 0.0],
                [0.0, 0.5, 0.5],
            ]
        )

        matrix = train.SAMDataInterface.create_interpolation_matrix(
            to_edges, from_edges
        )
        np.testing.assert_array_max_ulp(ref_values, matrix, maxulp=4)

    def test_interpolation_matrix_errors(self):
        from_grid = np.array([0.0, 1.0, 2.0, 3.0])
        to_grid = np.array([0.5, 1.5, 2.5])

        # to_grid not fully contained in from_grid
        with pytest.raises(ValueError):
            # Case below the minimum
            train.SAMDataInterface.create_interpolation_matrix(
                np.array([-1.0, 1.5, 2.5]), from_grid
            )
        with pytest.raises(ValueError):
            # Case above the maximum
            train.SAMDataInterface.create_interpolation_matrix(
                np.array([0.5, 1.5, 4.0]), from_grid
            )

        # Non-monotonic grids
        with pytest.raises(ValueError):
            # to_grid
            train.SAMDataInterface.create_interpolation_matrix(
                np.array([0.5, 1.5, 1.0]), from_grid
            )
        with pytest.raises(ValueError):
            # from_grid
            train.SAMDataInterface.create_interpolation_matrix(
                to_grid, np.array([0.0, 2.0, 1.0, 3.0])
            )
        with pytest.raises(ValueError):
            # to_grid with NaN (is not sorted)
            train.SAMDataInterface.create_interpolation_matrix(
                np.array([0.5, np.nan, 2.5]), from_grid
            )
        with pytest.raises(ValueError):
            # from_grid with NaN (is not sorted)
            train.SAMDataInterface.create_interpolation_matrix(
                to_grid, np.array([0.0, 1.0, np.nan, 3.0])
            )

        # Degenerate grids
        with pytest.raises(ValueError):
            # to_grid
            train.SAMDataInterface.create_interpolation_matrix(
                np.array([0.5]), from_grid
            )
        with pytest.raises(ValueError):
            # from_grid
            train.SAMDataInterface.create_interpolation_matrix(to_grid, np.array([0.0]))
