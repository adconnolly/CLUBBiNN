"""Simple smoke tests on the datasets under data/."""

import pytest


class TestBOMEX:
    """
    Simple checks on the (reduced) BOMEX dataset.

    In reality provided the md5 checksum in the bomex_dataset fixture passes
    these should all be successful.
    """

    def test_variables(self, bomex_dataset):
        """Check there are 16 variables."""
        assert len(bomex_dataset.data_vars) == 16

    def test_dimensions(self, bomex_dataset):
        """Check the expected dimensions are present."""
        expected_dims = {"time", "x", "y", "z"}
        assert expected_dims.issubset(set(bomex_dataset.dims)), (
            f"Missing dimensions: {expected_dims - set(bomex_dataset.dims)}"
        )

    def test_dimension_sizes(self, bomex_dataset):
        """Check the dimensions are of the correct sizes."""
        expected_dims = {
            "time": 120,
            "x": 1,
            "y": 1,
            "z": 75,
        }
        for dim, expected_size in expected_dims.items():
            assert bomex_dataset.sizes[dim] == expected_size, (
                f"{dim} has size {bomex_dataset.dims[dim]}, expected {expected_size}"
            )
