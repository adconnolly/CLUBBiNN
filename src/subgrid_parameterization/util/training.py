"""Utility functions for training models for C14 prediction."""

import torch
import xarray as xr
import warnings
from pathlib import Path

import subgrid_parameterization.preprocess.SAM_helpers as sam
import subgrid_parameterization.preprocess.mixing_length as mixing_length

from subgrid_parameterization.preprocess.C14 import get_C14

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

    _zm: npt.NDArray
    _zt: npt.NDArray

    @staticmethod
    def convert_sam_grid_to_clubb(
        z_sam: npt.ArrayLike,
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Construct CLUBB grids from the SAM LES grid:

        Parameters
        ----------
        z_sam : numpy.array_like
            Vertical grid from SAM LES results [meters]
            In principle can have a rank 1 or 2 (when grid varies with time).
            However only rank 1 is implemented at the moment.

        Returns
        -------
        numpy.ndarray
            zm : CLUBB Momentum grid [meters]
        numpy.ndarray
            zt : CLUBB Thermodynamic grid [meters]
        """

        # Ensure that we will return the correct type
        z_sam = np.asarray(z_sam)

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
        self._sam_dataset = sam_dataset

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
    def zm(self) -> npt.NDArray:
        """CLUBB Momentum grid [meters]"""
        return self._zm

    @property
    def zt(self) -> npt.NDArray:
        """CLUBB Thermodynamic grid [meters]"""
        return self._zt

    @staticmethod
    def interpolate_with_extrapolation(
        x_new: npt.NDArray, x: npt.NDArray, y: npt.NDArray
    ) -> npt.NDArray:
        """Piece-wise linear interpolate y(x) to new grid with extrapolation.

        Parameters
        ----------
        x_new : numpy.ndarray
            New grid
        x : numpy.ndarray
            Original grid that must be 1D array
        y : numpy.ndarray
            Original y values.

        Returns
        -------
        numpy.ndarray
            Interpolated (or extrapolated) y values at x_new.
        """

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
    ) -> npt.NDArray:
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
            self._sam_dataset["z"].values,
            self._sam_dataset[varname],
        )

    def get_sam_pressure_on_clubb_grid(self, grid_type: str) -> npt.NDArray:
        """Get the SAM pressure variable interpolated on the CLUBB grid.

        Parameters
        ----------
        grid : str
            Target grid for interpolation. Must be either 'zm' or 'zt'.

        Returns
        -------
        numpy.ndarray
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
            self._sam_dataset["z"].values,
            self._sam_dataset["p"].values,
        )
