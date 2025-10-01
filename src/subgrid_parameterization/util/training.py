"""Utility functions for training models for C14 prediction."""

import xarray as xr
import warnings
import typing
from pathlib import Path

import numpy.typing as npt
import numpy as np

from scipy.interpolate import interp1d


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


class SAMDataInterface:
    """Wraps around SAM-LES results to expose data in a CLUBB-like fashion.

    SAM in System for Atmospheric Modelling [http://rossby.msrc.sunysb.edu/SAM.html] # TODO: Verify this

    Since CLUBB is using a staggered grid, but the LES is not, some variables
    must be represented in a different location (on an offset grid).

    CLUBB has two grids in a single column [https://arxiv.org/pdf/1711.03675v1#nameddest=url:clubb_grid]:
     - `zm`: Momentum grid
     - `zt`: Thermodynamic grid

    #TODO: Document the logic of how LES grid is converted to CLUBB grids Here.
    #      I fear I do not understand exactly why it is done like it is

    SAM variables exist in three flavours:
      - Time invariant pressure profile (dims: ("z"))
      - Time-series scalars (dims: ("time"))
      - Variables (dims: ("time", "z", "y", "x"))
    Only datasets with degenerate y and x dimensions are supported.
    """

    _sam_dataset: xr.Dataset

    _zm: npt.NDArray[np.float64]
    _zt: npt.NDArray[np.float64]

    @staticmethod
    def convert_sam_grid_to_clubb(
        z_sam: npt.ArrayLike,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Construct CLUBB grids from the SAM LES grid:

        Parameters
        ----------
        z_sam : numpy.array_like
            Vertical grid from SAM LES results [meters]
            In principle can have a rank 1 or 2 (when grid varies with time).
            However only rank 1 is implemented at the moment.

        Returns
        -------
        numpy.ndarray[np.float64]
            zm : CLUBB Momentum grid [meters]
        numpy.ndarray[np.float64]
            zt : CLUBB Thermodynamic grid [meters]
        """

        # Ensure that we will return the correct type
        z_sam = np.asarray(z_sam, dtype=np.float64)

        if len(z_sam.shape) != 1:
            # OK, indexing with Ellipsis below should take care of rank 2 (and higher) case
            # but remove error only after thorough testing
            raise NotImplementedError(
                "Time-varying grid is not yet supported. SAM grid must be rank 1 at the moment."
            )

        # TODO: The code does exactly what it did before, but we need to
        #   document the motivation behind what it is doing.
        #   e.g. why is the 0th element ignored
        zm = 0.5 * (z_sam[..., 1:-1:2] + z_sam[..., 2::2])
        zm = np.insert(zm, 0, 0.0, axis=-1)  # Add surface level at the bottom
        zt = (zm[..., :-1] + zm[..., 1:]) / 2
        return zm, zt

    def __init__(self, sam_dataset: xr.Dataset):
        """Initialize the data interface.

        Parameters
        ----------
        sam_dataset : xarray.Dataset
            Dataset containing SAM-LES results.
        """
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

        # Convert the SAM grid to CLUBB grids
        self._zm, self._zt = self.convert_sam_grid_to_clubb(sam_dataset["z"].values)

        # Paranoid people make their read-only data read-only
        self._zm.setflags(write=False)
        self._zt.setflags(write=False)

    @property
    def zm(self) -> npt.NDArray[np.float64]:
        """CLUBB Momentum grid [meters]"""
        return self._zm

    @property
    def zt(self) -> npt.NDArray[np.float64]:
        """CLUBB Thermodynamic grid [meters]"""
        return self._zt

    T = typing.TypeVar("T", bound=np.floating)

    @staticmethod
    def create_interpolation_matrix(
        to_grid: npt.NDArray[T], from_grid: npt.NDArray[T]
    ) -> npt.NDArray[T]:
        """Create piece-wise constant interpolation matrix between two 1d grids.

        Parameters
        ----------
        to_grid : 1D numpy.ndarray
            Boundaries between the cells of the target grid.
        from_grid : 1D numpy.ndarray
            Boundaries between the cells of the source grid.

        Returns
        -------
        2D (N-1)x(M-1) numpy.ndarray
        where `len(to_grid)` is N and `len(from_grid)` M
        Interpolation matrix TODO: Finish this

        Note
        ----
        Data storage on both grids is assumed as average values of the variable
        at cell centres. Hence, since inputs are cell boundaries. The number of
        values stored on from_grid on len(from_grid) == N is N-1.

        When calculating averaged values on the target grid, we also assume that
        these are average values at cell centres.
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

        # Build the matrix element by element
        #
        # This is the easiest conceptually way to code it!
        #
        # TODO: REFACTOR THIS INEFFICIENT MONSTROSITY
        matrix = np.zeros([len(to_grid) - 1, len(from_grid) - 1])
        for i_col in range(len(to_grid) - 1):
            for i_row in range(len(from_grid) - 1):
                x_b, x_t = from_grid[i_row : i_row + 2]
                y_b, y_t = to_grid[i_col : i_col + 2]

                if y_b == y_t:
                    # Degenerate cell in target grid
                    matrix[i_col, i_row] = 0.0
                    continue

                matrix[i_col, i_row] = max(0.0, min(x_t, y_t) - max(x_b, y_b)) / (
                    y_t - y_b
                )
        return matrix

    @staticmethod
    def interpolate_with_extrapolation(
        x_new: npt.NDArray[T], x: npt.NDArray[T], y: npt.NDArray[T]
    ) -> npt.NDArray[T]:
        """Piece-wise linear interpolate y(x) to new grid with extrapolation.

        Parameters
        ----------
        x_new : numpy.ndarray[T]
            New grid
        x : numpy.ndarray[T]
            Original grid that must be 1D array
        y : numpy.ndarray[T]
            Original y values.

        Where T is some floating point dtype.

        Returns
        -------
        numpy.ndarray[T]
            Interpolated (or extrapolated) y values at x_new.
        """

        # Do not allow silent mixed precision computation
        # Check that dtype of all inputs matches
        if x_new.dtype != x.dtype or x.dtype != y.dtype:
            raise ValueError(
                "Mixed precision is not supported. Cast all inputs to same dtype before call. Was given: "
                f"x_new.dtype={x_new.dtype}, x.dtype={x.dtype}, y.dtype={y.dtype}"
            )

        # Make error in case of ND x array more explicit
        if len(x.shape) != 1:
            raise ValueError(f"Original x grid must be 1D array. Has shape: {x.shape}")

        # TODO: interp1d is 'Legacy' (but not deprecated). Perhaps this should be replaced.
        interpolator = interp1d(
            x,
            y,
            axis=-1,
            fill_value="extrapolate",
        )
        return interpolator(x_new)

    def get_sam_variable_on_clubb_grid(
        self, varname: str, grid_type: str
    ) -> npt.NDArray[np.float64]:
        """Get a SAM result variable interpolated on the CLUBB grid.

        Parameters
        ----------
        varname : str
            Name of the variable in the SAM dataset.
        grid : str
            Target grid for interpolation. Must be either 'zm' or 'zt'.

        Returns
        -------
        numpy.ndarray
            Interpolated variable on the target grid.
        """
        if grid_type not in ("zm", "zt"):
            raise ValueError(
                "SAM variable must be interpolated on either CLUBB momentum ('zm') or thermodynamic ('zt') grid."
            )

        if varname not in self._sam_dataset:
            raise ValueError(f"Variable '{varname}' not found in the dataset.")

        if {"time", "z"} != set(self._sam_dataset[varname].dims):
            raise ValueError(
                f"Variable '{varname}' must have dimensions ('time', 'z'). Has {self._sam_dataset[varname].dims}."
                "If you are trying to access pressure, use the dedicated method."
            )

        z = self._zm if grid_type == "zm" else self._zt

        return self.interpolate_with_extrapolation(
            z,
            np.asarray(self._sam_dataset["z"].values, dtype=np.float64),
            np.asarray(self._sam_dataset[varname].values, dtype=np.float64),
        )

    def get_sam_pressure_on_clubb_grid(self, grid_type: str) -> npt.NDArray[np.float64]:
        """Get the SAM pressure variable interpolated on the CLUBB grid.

        Parameters
        ----------
        grid : str
            Target grid for interpolation. Must be either 'zm' or 'zt'.

        Returns
        -------
        numpy.ndarray[np.float64]
            Interpolated pressure on the target grid.
        """
        if grid_type not in ("zm", "zt"):
            raise ValueError(
                "SAM pressure must be interpolated on either CLUBB momentum ('zm') or thermodynamic ('zt') grid."
            )

        if "p" not in self._sam_dataset:
            raise ValueError("Pressure variable 'p' not found in the dataset.")

        z = self._zm if grid_type == "zm" else self._zt

        return self.interpolate_with_extrapolation(
            z,
            np.asarray(self._sam_dataset["z"].values, dtype=np.float64),
            np.asarray(self._sam_dataset["p"].values, dtype=np.float64),
        )
