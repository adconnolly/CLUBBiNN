import xarray as xr
import numpy as np
import subgrid_parameterization.preprocess.saminterface as sam


def preprocess(files, data_root=""):
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

    input = list()
    output = list()
    aux_dict = {}
    Nsamples = [0]

    for file in files:
        ds = xr.open_dataset(data_root + file + ".nc")

        # Create a CLUBB momentum grid and the dataset
        z_sam = np.asarray(ds["z"], dtype=np.float64)
        nzm = (len(z_sam) + 1) // 2
        zm = np.concatenate(
            ([0], 0.5 * (z_sam[1 : 2 * nzm - 1 : 2] + z_sam[2 : 2 * nzm - 1 : 2]))
        )
        grids = sam.CLUBBGrids.from_momentum_grid(zm)
        sam_ds = sam.SAMDataInterface(ds, grids)

        ngrdcol = len(ds["time"])
        nzt = len(grids.zt)

        L, Lup, Ldown = sam_ds.get_mixing_length()
        Hscale = 1000  # 1km
        C14 = sam_ds.get_C14()
        up2 = sam_ds.get_sam_variable_on_clubb_grid("U2", "zt")
        vp2 = sam_ds.get_sam_variable_on_clubb_grid("V2", "zt")
        wp2 = sam_ds.get_sam_variable_on_clubb_grid("W2", "zt")
        e = 0.5 * (up2 + vp2 + wp2)
        disp = sam_ds.get_disp()
        C14min = 0.2
        C14max = 2
        minMask = disp < -2 / 3 * C14min / L * e**1.5
        maxMask = e > (-1.5 * disp * L / C14max) ** (2 / 3)

        for it in range(ngrdcol):
            for k in range(nzt):
                if minMask[it, k] and maxMask[it, k]:
                    input.append(
                        [
                            up2[it, k] / e[it, k],
                            vp2[it, k] / e[it, k],
                            wp2[it, k] / e[it, k],
                            Lup[it, k] / Hscale,
                            Ldown[it, k] / Hscale,
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
