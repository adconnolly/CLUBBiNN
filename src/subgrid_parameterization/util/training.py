"""Utility functions for training models for C14 prediction."""

import xarray as xr
import warnings
import typing
from pathlib import Path

import numpy.typing as npt
import numpy as np

from dataclasses import dataclass

import scipy.sparse

T_FLOAT = typing.TypeVar("T_FLOAT", bound=np.floating)


def read_as_xarray(path: Path | str) -> xr.Dataset:
    """
    Read a netCDF datafile as an xarray Dataset.

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


@dataclass(frozen=True)
class CLUBBGrids:
    """
    Pair of CLUBB like (momentum and thermodynamic) grids.

    Refer to:
    https://github.com/m2lines/clubb_ML/blob/92b8d7aeeafc1b045641b4c91806144a1c68945b/src/CLUBB_core/grid_class.F90
    for extra details on CLUBB grids.

    We assume that the functions are represented as average values over the cells
    and the grids are staggered such that the thermodynamic grid `zt` values are
    edges of the cells of the momentum grid and vice-versa.

    We do store the cell edges explicitly as separate arrays to allow us do deal
    with the problem of how to define edges at the top and bottom of the grid
    and we need to introduce one extra edge to either grid.

    To construct the grids use on of the provided factory methods.

    Attributes
    ----------
    zm : numpy.ndarray
        CLUBB momentum grid [meters].
    zt : numpy.ndarray
        CLUBB thermodynamic grid [meters].
    zm_cell_edges : numpy.ndarray
        Edges of the cells of the CLUBB momentum grid [meters].
    zt_cell_edges : numpy.ndarray
        Edges of the cells of the CLUBB thermodynamic grid [meters].
    """

    zm: npt.NDArray[np.float64]
    zt: npt.NDArray[np.float64]

    zm_cell_edges: npt.NDArray[np.float64]
    zt_cell_edges: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        """Make it really immutable."""
        self.zm.setflags(write=False)
        self.zt.setflags(write=False)
        self.zm_cell_edges.setflags(write=False)
        self.zt_cell_edges.setflags(write=False)

    @classmethod
    def from_momentum_grid(cls, zm: npt.NDArray[np.float64]) -> "CLUBBGrids":
        """
        Grid defined from momentum levels (CLUBB grid_type = 3).

        The thermodynamic levels are derived from the momentum levels as mid-points
        of the momentum grid cells. So if len(zm) is N, then len(zt) is N-1.

        Defining the momentum grid cells is a bit ambiguous, we take following
        assumptions:
          - The bottom edge is degenerate and starts at zm[0] (i.e. level stores
          average over the call but is placed at the start of the cell)
          - The top edge is defined such that the top momentum level is at
          the mid-point. Consequently thermodynamic and momentum grids span
          different ranges.

        Parameters
        ----------
        zm : numpy.ndarray
            Edges of the cells of the CLUBB momentum grid [meters]. Sorted,
            strictly increasing.
        """
        # Check preconditions of the CLUBB momentum grid
        zm = np.array(zm, dtype=np.float64)

        if not np.all(zm[1:] > zm[:-1]):
            raise ValueError("CLUBB momentum grid must strictly increasing.")

        if len(zm) < 2:
            raise ValueError(
                f"CLUBB momentum grid must have at least two levels. Got {len(zm)} levels."
            )

        # Thermodynamic grid parameters
        zt = (zm[:-1] + zm[1:]) / 2
        zt__cell_edges = zm

        # Define momentum grid cell edges
        top_zm_edge = zt[-1] + 2.0 * (zm[-1] - zt[-1])
        zm_cell_edges = np.concatenate([[zm[0]], zt, [top_zm_edge]])

        return cls(
            zm=zm, zt=zt, zm_cell_edges=zm_cell_edges, zt_cell_edges=zt__cell_edges
        )


class SAMDataInterface:
    """
    Wraps around SAM-LES results to expose data in a CLUBB-like fashion.

    SAM in System for Atmospheric Modelling [http://rossby.msrc.sunysb.edu/SAM.html]

    Since CLUBB is using a staggered grid, but the LES is not, some variables
    must be represented in a different location (on an offset grid).

    CLUBB has two grids in a single column [https://arxiv.org/pdf/1711.03675v1#nameddest=url:clubb_grid]:
     - `zm`: Momentum grid
     - `zt`: Thermodynamic grid

    SAM variables exist in three flavours:
      - Time invariant pressure profile (dims: ("z"))
      - Time-series scalars (dims: ("time"))
      - Variables (dims: ("time", "z", "y", "x"))
    Only datasets with degenerate y and x dimensions are supported.
    """

    _sam_dataset: xr.Dataset

    _grids: CLUBBGrids

    _z_to_zm_matrix: scipy.sparse.csr_array
    _z_to_zt_matrix: scipy.sparse.csr_array

    def __init__(self, sam_dataset: xr.Dataset, grids: CLUBBGrids) -> None:
        """
        Initialize the data interface.

        Parameters
        ----------
        sam_dataset : xarray.Dataset
            Dataset containing SAM-LES results.
        grids : CLUBBGrids
            CLUBB-like momentum and thermodynamic grids.
        """
        # Load the dataset
        self._sam_dataset = sam_dataset.copy(deep=True)

        # Verify that only expected coordinates are present in the dataset
        if {"time", "z", "y", "x"} != set(self._sam_dataset.coords):
            raise ValueError(
                f"Dataset must contain 'time', 'z', 'y', and 'x' coordinates. Has {self._sam_dataset.coords}"
            )

        # Verify degenerate dimensions and remove them
        if self._sam_dataset["y"].shape != (1,):
            raise ValueError(
                f"'y' dimension must be degenerate (shape: (1,)), but is: {self._sam_dataset['y'].shape}"
            )
        if self._sam_dataset["x"].shape != (1,):
            raise ValueError(
                f"'x' dimension must be degenerate (shape: (1,)), but is: {self._sam_dataset['x'].shape}"
            )
        self._sam_dataset = self._sam_dataset.squeeze()

        # Load the grids
        # They are immutable so no need to copy
        self._grids = grids

        # We need to infer the edges of the SAM grid cells from the mid-point values
        # We assume that the grid starts at ground level
        sam_z = np.asarray(self._sam_dataset["z"].values, dtype=np.float64)
        sam_z_edges = self.edges_from_midpoints(0.0, sam_z)

        # Store projection matrices from SAM grid to CLUBB grids
        self._z_to_zm_matrix = self.create_projection_matrix(
            grids.zm_cell_edges, sam_z_edges
        )
        self._z_to_zt_matrix = self.create_projection_matrix(
            grids.zt_cell_edges, sam_z_edges
        )

    @property
    def grids(self) -> CLUBBGrids:
        """Get the CLUBB-like grids."""
        return self._grids

    @property
    def sam_dataset(self) -> xr.Dataset:
        """Get the underlying SAM dataset."""
        return self._sam_dataset

    @staticmethod
    def edges_from_midpoints(
        start_point: np.floating, midpoints: npt.NDArray[T_FLOAT]
    ) -> npt.NDArray[T_FLOAT]:
        """
        Compute cell edges from mid-point values.

        Parameters
        ----------
        start_point : T_FLOAT
            Location of the start of the first (bottom) cell. Will be converted
            to the same dtype as midpoints, hence may shift slightly.
        midpoints : numpy.ndarray[T_FLOAT]
            Mid-point values of the cells.

        Returns
        -------
        numpy.ndarray[T_FLOAT]
            Cell edge values.
        """
        start_point = midpoints.dtype.type(start_point)

        # Empty midpoints
        if len(midpoints) == 0:
            raise ValueError("Midpoints array must not be empty.")

        # Invalid start point
        if start_point >= midpoints[0]:
            raise ValueError(
                f"start_point '{start_point}' must be less than the first midpoint value '{midpoints[0]}'."
            )

        # Check monotonicity of midpoints
        if not np.all(midpoints[1:] > midpoints[:-1]):
            raise ValueError("Midpoints must be strictly increasing.")

        # Do a loop for now and refactor later
        edges = [start_point]
        for mid_point in midpoints:
            last_edge = edges[-1]
            next_edge = 2.0 * mid_point - last_edge
            edges.append(next_edge)
        edges = np.array(edges, dtype=midpoints.dtype)

        # Throw error if the reconstruction is invalid
        # Grid is strictly increasing
        if not np.all(edges[1:] > edges[:-1]):
            raise ValueError(
                "Reconstruction has failed to produce a valid grid. Not increasing."
            )

        # Interior edges are between midpoints
        # TODO: Is this check redundant given the above? Probably. Get 2nd opinion during review!
        if not np.all(midpoints > edges[:-1]) or not np.all(midpoints < edges[1:]):
            raise ValueError(
                "Reconstruction has failed to produce a valid grid. Midpoints not between edges."
            )

        return edges

    @staticmethod
    def create_projection_matrix(
        to_grid: npt.NDArray[T_FLOAT], from_grid: npt.NDArray[T_FLOAT]
    ) -> scipy.sparse.csr_array:
        """
        Create a projection matrix between two piece-wise constant 1d grids.

        Data storage on both grids is assumed as average values of the variable
        at cell centres. Hence, since inputs are cell boundaries. The number of
        values stored on from_grid on len(from_grid) == N is N-1.

        Parameters
        ----------
        to_grid : 1D numpy.ndarray
            Boundaries between the cells of the target grid.
        from_grid : 1D numpy.ndarray
            Boundaries between the cells of the source grid.

        Returns
        -------
        2D (N-1)x(M-1) scipy.sparse.csr_array
            Where `len(to_grid)` is N and `len(from_grid)` M
            Projection matrix that maps values from the source grid to the target grid.
        """

        # Assert preconditions
        if to_grid[0] < from_grid[0] or to_grid[-1] > from_grid[-1]:
            raise ValueError(
                "to_grid must be fully contained in from_grid. "
                f"Got to_grid[0]={to_grid[0]}, from_grid[0]={from_grid[0]}, "
                f"to_grid[-1]={to_grid[-1]}, from_grid[-1]={from_grid[-1]}"
            )
        if len(to_grid.shape) != 1 or len(from_grid.shape) != 1:
            raise ValueError(
                "Both to_grid and from_grid must be 1D arrays. "
                f"Got to_grid.shape={to_grid.shape}, from_grid.shape={from_grid.shape}"
            )
        if len(to_grid) < 2 or len(from_grid) < 2:
            raise ValueError(
                "Both to_grid and from_grid must have at least two elements (one cell). "
                f"Got len(to_grid)={len(to_grid)}, len(from_grid)={len(from_grid)}"
            )

        # Note that the comparison here is not arbitrary
        # We want to make sure that the grid in not judged sorted if it contains NaN
        # (which is correct, since NaNs are not ordered)
        if not np.all(to_grid[1:] >= to_grid[:-1]):
            raise ValueError("'to_grid' must be sorted in ascending order.")
        if not np.all(from_grid[1:] >= from_grid[:-1]):
            raise ValueError("'from_grid' must be sorted in ascending order.")

        # Build the sparse matrix row by row
        # We use LIL for building since it is more efficient to modify
        # When the array is build, we switch to CSR to have faster matrix-vector product
        # operations
        matrix = scipy.sparse.lil_array(
            (len(to_grid) - 1, len(from_grid) - 1), dtype=to_grid.dtype
        )
        for i_col in range(len(to_grid) - 1):
            # Find index of upper and lowe bounds for non-zero elements of the matrix
            y_b, y_t = to_grid[i_col : i_col + 2]

            if y_b == y_t:
                # If the target cell is degenerate we can just go to the next row
                continue

            i_lb = np.searchsorted(from_grid, y_b, side="right") - 1
            i_ub = np.searchsorted(from_grid, y_t, side="left")

            x_b = from_grid[i_lb:i_ub]
            x_t = from_grid[i_lb + 1 : i_ub + 1]

            # Weight of a contribution of a `from` grid bin in a `target` bin
            # is just a length of the intersection between the bins.
            # We need to clip the length at 0.0 in case intersection is empty
            matrix[i_col, i_lb:i_ub] = np.maximum(
                0.0, np.minimum(x_t, y_t) - np.maximum(x_b, y_b)
            ) / (y_t - y_b)
        return scipy.sparse.csr_array(matrix)

    def get_sam_variable_on_clubb_grid(
        self, varname: str, grid_type: str
    ) -> npt.NDArray[np.float64]:
        """
        Get a SAM result variable projected to the CLUBB grid.

        Parameters
        ----------
        varname : str
            Name of the variable in the SAM dataset.
        grid : str
            Target grid for projection. Must be either 'zm' or 'zt'.

        Returns
        -------
        numpy.ndarray
            Projected variable on the target grid.
        """
        if grid_type not in ("zm", "zt"):
            raise ValueError(
                "SAM variable must be projected on either CLUBB momentum ('zm') or thermodynamic ('zt') grid."
            )

        if varname not in self._sam_dataset:
            raise ValueError(f"Variable '{varname}' not found in the dataset.")

        if {"time", "z"} != set(self._sam_dataset[varname].dims):
            raise ValueError(
                f"Variable '{varname}' must have dimensions ('time', 'z'). Has {self._sam_dataset[varname].dims}."
                "If you are trying to access pressure, use the dedicated method."
            )

        matrix = self._z_to_zm_matrix if grid_type == "zm" else self._z_to_zt_matrix

        sam_var = np.asarray(self._sam_dataset[varname].values, dtype=np.float64)

        # Perform the matrix multiplication for each time step
        return sam_var @ matrix.T

    def get_sam_pressure_on_clubb_grid(self, grid_type: str) -> npt.NDArray[np.float64]:
        """
        Get the SAM pressure variable projected to the CLUBB grid.

        Parameters
        ----------
        grid : str
            Target grid for projection. Must be either 'zm' or 'zt'.

        Returns
        -------
        numpy.ndarray[np.float64]
            Projected pressure on the target grid.
        """
        if grid_type not in ("zm", "zt"):
            raise ValueError(
                "SAM pressure must be projected to either CLUBB momentum ('zm') or thermodynamic ('zt') grid."
            )

        if "p" not in self._sam_dataset:
            raise ValueError("Pressure variable 'p' not found in the dataset.")

        matrix = self._z_to_zm_matrix if grid_type == "zm" else self._z_to_zt_matrix

        sam_p = np.asarray(self._sam_dataset["p"].values, dtype=np.float64)

        return matrix @ sam_p
