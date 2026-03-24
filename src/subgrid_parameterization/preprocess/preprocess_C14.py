import xarray as xr
import numpy as np
import subgrid_parameterization.preprocess.saminterface as sam


def preprocess(
    files, C14min=0.2, C14max=2, data_root="", filesStart=None, filesZtop=None
):
    """
    Loop through files and return network inputs and outputs.

    Parameters
    ----------
    files : List or str
    data_root : str
        String to directory where all the paths in files begin


    Returns
    -------
    numpy.ndarray[np.float64]
        Inputs to network
    numpy.ndarray[np.float64]
        Outputs to network
    """

    if filesStart == None:
        filesStart = len(files) * [0]

    input = list()
    output = list()
    aux_dict = {}
    Nsamples = [0]

    for ifile, file in enumerate(files):
        ds = xr.open_dataset(data_root + file + ".nc")

        # Create a CLUBB momentum grid and the dataset
        z_sam = np.asarray(ds["z"], dtype=np.float64)
        nzm = (len(z_sam) + 1) // 2
        if np.isclose(2.0 * z_sam[0], z_sam[1]):
            zm = np.concatenate(([0], z_sam[2 : 2 * nzm - 1 : 2]))
            print("Combined staggered grids")
        else:
            zm = np.concatenate(
                ([0], 0.5 * (z_sam[1 : 2 * nzm - 1 : 2] + z_sam[2 : 2 * nzm - 1 : 2]))
            )
        grids = sam.CLUBBGrids.from_momentum_grid(zm)
        sam_ds = sam.SAMDataInterface(ds, grids)

        ngrdcol = len(ds["time"])
        itStart = filesStart[ifile]

        if filesZtop == None:
            ktop = len(grids.zm)
        else:
            if grids.zm[-1] > filesZtop[ifile]:
                ktop = np.searchsorted(grids.zm, filesZtop[ifile], side="right")
            else:
                ktop = len(grids.zm)

        L, Lup, Ldown = sam_ds.get_mixing_length()
        L_zm = np.array(
            [np.interp(grids.zm, grids.zt, L[icol]) for icol in range(ngrdcol)]
        )
        Lup_zm = np.array(
            [np.interp(grids.zm, grids.zt, Lup[icol]) for icol in range(ngrdcol)]
        )
        Ldown_zm = np.array(
            [np.interp(grids.zm, grids.zt, Ldown[icol]) for icol in range(ngrdcol)]
        )
        Hscale = 1000  # 1km
        C14 = sam_ds.get_C14()
        up2 = sam_ds.get_sam_variable_on_clubb_grid("U2", "zm")
        vp2 = sam_ds.get_sam_variable_on_clubb_grid("V2", "zm")
        wp2 = sam_ds.get_sam_variable_on_clubb_grid("W2", "zm")
        e = 0.5 * (up2 + vp2 + wp2)
        disp = sam_ds.get_disp()

        minMask = disp < -2 / 3 * C14min / L_zm * e**1.5
        maxMask = e > (-1.5 * disp * L_zm / C14max) ** (2 / 3)

        for it in range(itStart, ngrdcol):
            for k in range(1, ktop):
                if minMask[it, k] and maxMask[it, k]:
                    input.append(
                        [
                            up2[it, k] / e[it, k],
                            vp2[it, k] / e[it, k],
                            wp2[it, k] / e[it, k],
                            Lup_zm[it, k] / Hscale,
                            Ldown_zm[it, k] / Hscale,
                        ]
                    )
                    output.append([C14[it, k]])

        Nsamples.append(len(input))
        print(file)
        print(str(Nsamples[-1] - Nsamples[-2]) + " samples \n")

    aux_dict["Nsamples"] = [
        Nsamples[i] - Nsamples[i - 1] for i in range(1, len(files) + 1)
    ]

    return input, output, aux_dict


# def calc_zt2zm_weights(grids: CLUBBGrids) -> Array:
#     """
#     Calculate interpolation weights from thermodynamic to momentum levels.

#     Translated from calc_zt2zm_weights (grid_class.F90, lines 2027-2314).

#     Returns:
#         weights: shape (ngrdcol, nzm, 2)
#             weights[:, k, 0] = weight for upper zt level
#             weights[:, k, 1] = weight for lower zt level
#     """
#     zm = grids.zm
#     zt = grids.zt
#     zm_edges = grids.zm_cell_edges
#     dzm = zm_edges[1:] - zm_edges[:-1]

#     weights = np.zeros((len(zm), 2))

#     # # Interior momentum levels (k=1 to nzm-2)
#     # # Linear interpolation from surrounding thermodynamic levels
#     # def calc_interior_weight(k, w):
#     #     # Distance from lower zt to zm
#     #     dist_lower = jnp.abs(zm[:, k] - zt[:, k-1])
#     #     # Total distance between zt levels
#     #     total_dist = dzm[:, k] + 1e-30  # Avoid division by zero

#     #     # Weight for upper zt level (k-1 in 0-indexed)
#     #     w_upper = 1.0 - dist_lower / total_dist
#     #     # Weight for lower zt level (k in 0-indexed, but actually k-1 for zt)
#     #     w_lower = dist_lower / total_dist

#     #     w = w.at[:, k, T_ABOVE].set(w_upper)
#     #     w = w.at[:, k, T_BELOW].set(w_lower)
#     #     return w

#     # weights = lax.fori_loop(1, nzm - 1, calc_interior_weight, weights)

#     # # Boundary levels (extrapolation)
#     # # Lower boundary (k=0)
#     # weights = weights.at[:, 0, T_ABOVE].set(1.0)
#     # weights = weights.at[:, 0, T_BELOW].set(0.0)

#     # # Upper boundary (k=nzm-1)
#     # weights = weights.at[:, -1, T_ABOVE].set(1.0)
#     # weights = weights.at[:, -1, T_BELOW].set(0.0)

#     # return weights


#     # gr%weights_zt2zm(i,k,t_above) = ( gr%zm(i,k) - gr%zt(i,k-1) ) &
#     #                                     / ( gr%zt(i,k) - gr%zt(i,k-1) )
