"""
Code to reduce size and thin the BOMEX dataset.

Aims to provide a sample that code functionalities can be tested on.
Full datasets for training are large, and stored outside this repository.

This code:
- Drops variables not expected to be used by `stagger_var`
- Subsamples in time
- Aims to preserve coverage of the sample space/covariance of variables

It was originally run on BOMEX_64x64x75_100m_40m_1s which had 360 time samples and
450 variables to reduce it to 18 variables
"""

import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def thin_netcdf(
    input_path: str,
    output_path: str,
    variables_whitelist: list[str],
    coord_vars: list[str],
) -> None:
    """Thin dataset to whitelisted variables/coordinates and saves to a new file."""
    ds = xr.open_dataset(input_path)

    # Ensure all requested variables are present
    coords_to_keep = [c for c in coord_vars if c in ds.variables or c in ds.coords]
    vars_to_keep = [v for v in variables_whitelist if v in ds.variables]

    # Combine and deduplicate
    all_to_keep = list(dict.fromkeys(vars_to_keep + coords_to_keep))

    ds_sel = ds[all_to_keep]

    ds_sel.to_netcdf(output_path)


def thin_time(
    ds_path: str,
    output_path: str,
    drop_first: int = 0,
    time_thin_factor: int = 2,
) -> None:
    """
    Thin the dataset in the time dimension save the result.

    The dataset can be thinned by dropping a number of points from the start (spinup)
    and by reducing the number of samples in the remaining set by uniform sampling.

    Parameters
    ----------
    ds_path : str
        Path to the NetCDF file.
    output_path : str
        Path for the thinned NetCDF file.
    drop_first : int
        Number of initial time points to drop.
    time_thin_factor : int
        Keep every Nth time sample after dropping.
    """
    ds = xr.open_dataset(ds_path)

    if "time" in ds.dims:
        ds_thin = ds.isel(time=slice(drop_first, None, time_thin_factor))
    else:
        raise ValueError("No 'time' dimension found in dataset.")

    ds_thin.to_netcdf(output_path)
    print(f"Thinned dataset saved to {output_path}")


def generate_pairplot_and_cov(
    ds: xr.Dataset, variables: list[str], output_prefix: str
) -> None:
    """
    Generate seaborn pairplot for dataset.

    Uses an xarray.Dataset.
    Flattens all spatial and time dimensions as concerned with grid-box operations.
    Generates seaborn pairplot and covariance matrix for variables.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to analyze.
    variables : list of str
        List of variable names to include in the analysis.
    output_prefix : str
        Prefix for output files (e.g., 'before_subsampling').
    """
    # Only keep variables present in the dataset
    vars_present = [v for v in variables if v in ds.variables]

    # Flatten all dims for each variable and build a DataFrame
    data = {}
    for v in vars_present:
        arr = ds[v].values
        data[v] = arr.flatten()

    df = pd.DataFrame(data)
    df = df.dropna()

    # Pairplot
    sns.pairplot(df, plot_kws={"s": 2, "alpha": 0.3})
    plt.suptitle("Pairplot of variables (all grid points, all times)", y=1.02)
    plt.savefig(f"{output_prefix}_pairplot.png", bbox_inches="tight", dpi=200)
    plt.close()
    print(f"Pairplot saved to {output_prefix}_pairplot.png")

    # Covariance matrix
    cov = df.cov()
    cov.to_csv(f"{output_prefix}_cov_matrix.csv")
    print(f"Covariance matrix saved to {output_prefix}_cov_matrix.csv")


def print_variable_stats(ds: xr.Dataset, variables: list[str], label: str) -> None:
    """
    Print min, max, mean, and std for each variable in the dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to analyze.
    variables : list of str
        List of variable names to analyze.
    label : str
        Label for the stats output.
    """
    vars_present = [v for v in variables if v in ds.variables]
    data = {}

    for v in vars_present:
        arr = ds[v].values
        data[v] = arr.flatten()
    df = pd.DataFrame(data).dropna()

    print(f"\nStats for {label}:")
    for v in vars_present:
        vmin = df[v].min()
        vmax = df[v].max()
        mean = df[v].mean()
        std = df[v].std()
        print(
            f"{v:<10s} |"
            f" min: {vmin:13.6g} |"
            f" max: {vmax:13.6g} |"
            f" mean: {mean:13.6g} |"
            f" std: {std:13.6g}"
        )


def compare_covariances(
    cov_before_path: str,
    cov_after_path: str,
    output_path: str = "covariance_relative_change.csv",
    threshold: float = 0.05,
) -> None:
    """
    Read in two covariance matrices and write out the relative change.

    Print entries where the absolute relative change exceeds the threshold.

    Parameters
    ----------
    cov_before_path : str
        Path to the CSV file with the covariance matrix before thinning.
    cov_after_path : str
        Path to the CSV file with the covariance matrix after thinning.
    output_path : str
        Path to write the relative change CSV.
    threshold : float
        Only print entries where the absolute relative change exceeds this value (as a fraction, e.g., 0.05 for 5%).
    """
    cov_before = pd.read_csv(cov_before_path, index_col=0)
    cov_after = pd.read_csv(cov_after_path, index_col=0)

    # Compute relative change: (after - before) / before
    rel_change = (cov_after - cov_before) / cov_before.replace(0, np.nan)
    rel_change = rel_change.replace([np.inf, -np.inf], np.nan)

    rel_change.to_csv(output_path)
    print(f"Relative covariance change written to {output_path}\n")

    # Print only entries above threshold as percentage
    print(
        f"Relative covariance changes greater than {threshold * 100:.1f}% (as percentage):"
    )
    mask = rel_change.abs() > threshold
    any_printed = False
    for row, col in zip(*mask.values.nonzero()):
        val = rel_change.iloc[row, col]
        print(
            f"{rel_change.index[row]:<10s} vs "
            f"{rel_change.columns[col]:<10s}: {val * 100:8.2f}%"
        )
        any_printed = True
    if not any_printed:
        print("No relative changes above threshold.")


if __name__ == "__main__":
    VARIABLES_WHITELIST = [
        "U",
        "V",
        "WM",
        "U2",
        "V2",
        "W2",
        "U2DFSN",
        "V2DFSN",
        "W2DFSN",
        "THETA",
        "THETAV",
        "THETAL",
        "RTM",
        "THLM",
        "TABS",
    ]
    COORD_VARS = ["x", "y", "z", "time", "p"]

    original_dataset = "BOMEX_64x64x75_100m_40m_1s.nc"
    pruned_dataset = "BOMEX_pruned.nc"
    thinned_dataset = "BOMEX_pruned_thinned.nc"

    # Thin to variables/coords of interest
    thin_netcdf(
        input_path=original_dataset,
        output_path=pruned_dataset,
        variables_whitelist=VARIABLES_WHITELIST,
        coord_vars=COORD_VARS,
    )

    # Check data before thinning in time
    with xr.open_dataset(pruned_dataset) as ds:
        generate_pairplot_and_cov(
            ds,
            VARIABLES_WHITELIST,
            "before_subsampling",
        )
        print_variable_stats(
            ds,
            VARIABLES_WHITELIST,
            "before_subsampling",
        )

    # Thin in time
    thin_time(
        ds_path=pruned_dataset,
        output_path=thinned_dataset,
        drop_first=120,
        time_thin_factor=2,
    )

    # Check data after thinning in time
    with xr.open_dataset(thinned_dataset) as ds:
        generate_pairplot_and_cov(
            ds,
            VARIABLES_WHITELIST,
            "after_subsampling",
        )
        print_variable_stats(
            ds,
            VARIABLES_WHITELIST,
            "after_subsampling",
        )
    compare_covariances(
        cov_before_path="before_subsampling_cov_matrix.csv",
        cov_after_path="after_subsampling_cov_matrix.csv",
        output_path="covariance_relative_change.csv",
    )
