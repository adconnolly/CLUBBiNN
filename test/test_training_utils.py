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


@pytest.fixture
def clubb_like_grids():
    """Provides a CLUBB-like grid for testing."""
    zm = np.linspace(0.0, 1000.0, 51)
    return train.CLUBBGrids.from_momentum_grid(zm)


class TestSAMDataInterface:
    def test_init_errors(self, bomex_dataset, clubb_like_grids):
        ds = bomex_dataset

        # Missing coordinates
        coords_to_remove = ["y", "x", "z", "time"]
        for coord in coords_to_remove:
            with pytest.raises(ValueError):
                train.SAMDataInterface(ds.drop_vars(coord), clubb_like_grids)

        # Non-degenerate y or x dimensions
        for dim in ["y", "x"]:
            # We expand the dataset along a dimension
            temp = ds.copy(deep=True)
            temp.coords[dim].values[0] = ds.coords[dim].values[0] + 1.0
            combined = xr.concat([ds, temp], dim=dim, data_vars="minimal")

            with pytest.raises(ValueError):
                train.SAMDataInterface(combined, clubb_like_grids)

    def test_init(self, bomex_dataset, clubb_like_grids):
        # Just a smoke test
        # TODO: Figure out something more sensible
        sam_data = train.SAMDataInterface(bomex_dataset, clubb_like_grids)

        sam_data.grids.zm
        sam_data.grids.zt
        sam_data.sam_dataset

    @pytest.mark.parametrize("var_name", ["U", "V"])
    def test_projection(self, bomex_dataset, var_name):
        """Use coarsen SAM grid as a target.

        Basically we don't need to test that the projection matrix is correct
        since it is the role of other tests. Here we just want to assert that
        we use it correctly.

        We want target CLUBB-like cells to fit nicely inside SAM cells to
        make reference averaging easy.
        """
        z_sam = np.asarray(bomex_dataset["z"], dtype=np.float64)
        nzm = (len(z_sam) + 1) // 2
        zm = np.concatenate(
            ([0], 0.5 * (z_sam[1 : 2 * nzm - 1 : 2] + z_sam[2 : 2 * nzm - 1 : 2]))
        )
        grids = train.CLUBBGrids.from_momentum_grid(zm)

        # Compare against the reference"
        sam_data = train.SAMDataInterface(bomex_dataset, grids)
        var_zm = sam_data.get_sam_variable_on_clubb_grid(var_name, grid_type="zm")
        var_zt = sam_data.get_sam_variable_on_clubb_grid(var_name, grid_type="zt")

        # Compute reference by averaging
        sam_var = np.asarray(bomex_dataset[var_name], dtype=np.float64).squeeze()

        # Note that for the momentum grid the fist cell is only half, hence
        # it contains only a single SAM cell
        var_zm_ref = np.empty_like(var_zm)
        var_zm_ref[:, 0] = sam_var[:, 0]
        var_zm_ref[:, 1:] = 0.5 * (sam_var[:, 1:-1:2] + sam_var[:, 2::2])

        np.testing.assert_array_almost_equal_nulp(var_zm, var_zm_ref, nulp=4)

        # For the thermodynamic grid we just don't have the top SAM cell in
        # the range
        var_zt_ref = 0.5 * (sam_var[:, 0:-2:2] + sam_var[:, 1:-1:2])
        np.testing.assert_array_almost_equal_nulp(var_zt, var_zt_ref, nulp=4)

    def test_projection_pressure(self, bomex_dataset):
        """ "
        See `test_projection` for details.

        Pressure has it own dedicated method but logic is the same.
        """
        z_sam = np.asarray(bomex_dataset["z"], dtype=np.float64)
        nzm = (len(z_sam) + 1) // 2
        zm = np.concatenate(
            ([0], 0.5 * (z_sam[1 : 2 * nzm - 1 : 2] + z_sam[2 : 2 * nzm - 1 : 2]))
        )
        grids = train.CLUBBGrids.from_momentum_grid(zm)

        sam_data = train.SAMDataInterface(bomex_dataset, grids)
        pressure_zm = sam_data.get_sam_pressure_on_clubb_grid(grid_type="zm")
        pressure_zt = sam_data.get_sam_pressure_on_clubb_grid(grid_type="zt")

        # Compute reference by averaging
        sam_p = np.asarray(bomex_dataset["p"], dtype=np.float64).squeeze()

        pressure_zm_ref = np.empty_like(pressure_zm)
        pressure_zm_ref[0] = sam_p[0]
        pressure_zm_ref[1:] = 0.5 * (sam_p[1:-1:2] + sam_p[2::2])
        np.testing.assert_array_almost_equal_nulp(pressure_zm, pressure_zm_ref, nulp=4)

        pressure_zt_ref = 0.5 * (sam_p[0:-2:2] + sam_p[1:-1:2])
        np.testing.assert_array_almost_equal_nulp(pressure_zt, pressure_zt_ref, nulp=4)

    def test_projection_errors(self, bomex_dataset):
        z_sam = np.asarray(bomex_dataset["z"], dtype=np.float64)
        nzm = (len(z_sam) + 1) // 2
        zm = np.concatenate(
            ([0], 0.5 * (z_sam[1 : 2 * nzm - 1 : 2] + z_sam[2 : 2 * nzm - 1 : 2]))
        )
        grids = train.CLUBBGrids.from_momentum_grid(zm)

        sam_data = train.SAMDataInterface(bomex_dataset, grids)

        # Invalid grid type
        with pytest.raises(ValueError):
            sam_data.get_sam_variable_on_clubb_grid("U", grid_type="invalid_grid")
        with pytest.raises(ValueError):
            sam_data.get_sam_pressure_on_clubb_grid(grid_type="invalid_grid")

        # Invalid variable name
        with pytest.raises(ValueError):
            sam_data.get_sam_variable_on_clubb_grid("Not is dataset", grid_type="zm")

        # Accessing pressure via generic interface
        with pytest.raises(ValueError):
            sam_data.get_sam_variable_on_clubb_grid("p", grid_type="zm")


class TestSAMDataInterface_midpoint_to_edges:
    """Verify that the reverse computation of cell edges from mid-point values is correct."""

    @pytest.mark.parametrize("seed", [42, 767, 867, 8975])
    @pytest.mark.parametrize("max_z", [10.0, 100.0])
    @pytest.mark.parametrize("n_edges", [3, 10])
    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_on_random_grid(self, seed, max_z, n_edges, dtype):
        """Round trip test on random but sensible input"""
        prng = np.random.default_rng(seed)

        # We create some random edges but make sure to start at 0!
        edges = np.sort(
            np.concatenate(
                [[dtype(0.0)], prng.random(n_edges, dtype=dtype) * dtype(max_z)]
            )
        )
        mid_points = 0.5 * (edges[:-1] + edges[1:])
        edges_reconstructed = train.SAMDataInterface.edges_from_midpoints(
            0.0, mid_points
        )

        assert edges_reconstructed.dtype == dtype
        np.testing.assert_array_almost_equal_nulp(edges, edges_reconstructed, nulp=4)

    def test_errors(self):
        """Test that invalid inputs are rejected."""
        # At least one mid-point is required
        with pytest.raises(ValueError):
            train.SAMDataInterface.edges_from_midpoints(
                0.0, np.array([], dtype=np.float64)
            )

        # Inconsistent starting edge
        with pytest.raises(ValueError):
            train.SAMDataInterface.edges_from_midpoints(
                0.0, np.array([1.0, 1.9], dtype=np.float64)
            )

        # Non-monotonic mid-point values
        with pytest.raises(ValueError):
            train.SAMDataInterface.edges_from_midpoints(
                0.0, np.array([1.0, 2.0, 1.5], dtype=np.float64)
            )


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
        # Numpy test assertions do not work with sparse arrays
        np.testing.assert_array_max_ulp(ref_values, matrix.todense(), maxulp=4)

    def test_interpolation_matrix_to_degenerate_interval(self):
        from_edges = np.array([0.0, 1.0, 2.0, 3.0])
        to_edges = np.array([1.5, 1.5])

        ref_values = np.array([[0.0, 0.0, 0.0]])

        matrix = train.SAMDataInterface.create_interpolation_matrix(
            to_edges, from_edges
        )
        # Numpy test assertions do not work with sparse arrays
        np.testing.assert_array_max_ulp(ref_values, matrix.todense(), maxulp=4)

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
        # Numpy test assertions do not work with sparse arrays
        np.testing.assert_array_max_ulp(ref_values, matrix.todense(), maxulp=4)

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


class TestCLUBBGrids:
    """Test the interface for construction of the CLUBB-like grids."""

    def test_from_momentum_grid(self):
        zm = np.array([0.0, 1.0, 2.2, 3.0])
        grids = train.CLUBBGrids.from_momentum_grid(zm)

        def do_verification(grids, zm):
            zt_expected = np.array([0.5, 1.6, 2.6])
            zt_edges = np.array([0.0, 1.0, 2.2, 3.0])

            zm_edges = np.array([0.0, 0.5, 1.6, 2.6, 3.4])

            np.testing.assert_array_equal(grids.zt, zt_expected)
            np.testing.assert_array_equal(grids.zt_cell_edges, zt_edges)

            np.testing.assert_array_equal(grids.zm, zm)
            np.testing.assert_array_equal(grids.zm_cell_edges, zm_edges)

        do_verification(grids, zm)

        # Assert we are not aliasing anything
        # We want to make sure we cannot modify the grids via the input array
        zm_copy = zm.copy()
        zm[0] = 10.0
        do_verification(grids, zm_copy)

        # Make sure the arrays are really immutable
        with pytest.raises(ValueError):
            grids.zm[0] = 0.0
        with pytest.raises(ValueError):
            grids.zt[0] = 0.0
        with pytest.raises(ValueError):
            grids.zm_cell_edges[0] = 0.0
        with pytest.raises(ValueError):
            grids.zt_cell_edges[0] = 0.0

    def test_from_momentum_grid_errors(self):
        # Need at least two momentum levels to define thermodynamic levels
        zm = np.array([0.0])
        with pytest.raises(ValueError):
            train.CLUBBGrids.from_momentum_grid(zm)

        # Non-monotonic grid
        zm = np.array([0.0, 2.0, 1.0])
        with pytest.raises(ValueError):
            train.CLUBBGrids.from_momentum_grid(zm)

        # Grid with NaN
        zm = np.array([0.0, np.nan, 2.0])
        with pytest.raises(ValueError):
            train.CLUBBGrids.from_momentum_grid(zm)

    def test_regression_from_momentum_grid(self, reference_z_grids):
        _, zm_ref, zt_ref = reference_z_grids

        grids = train.CLUBBGrids.from_momentum_grid(zm_ref)
        np.testing.assert_array_equal(grids.zm, zm_ref)
        np.testing.assert_array_equal(grids.zt, zt_ref)
