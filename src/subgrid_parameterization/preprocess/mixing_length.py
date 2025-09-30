"""Mixing length-scales calculation based on CLUBB.

Provides interface for computing mixing length using the 5th moist, nonlocal length scale method
which is used by CLUBB.

Attributes
----------
    CLUBB_STANDALONE_CONSTANTS : dict[str, float]
        Dictionary that contains values of constants used in mixing length computation.
        They were logged during the standalone run of CLUBB. Note that, when embedded in
        a host model, the constants may be different. Keys match the names used internally
        in CLUBB.
"""

import numpy as np
import enum
import typing
import numpy.typing as npt
from subgrid_parameterization.preprocess import SAM_helpers as sam


FLOAT_T = typing.TypeVar("FLOAT_T", float, np.floating)


CLUBB_STANDALONE_CONSTANTS: dict[str, float] = {
    "Cp": 0.10046700000000000e04,
    "Rd": 0.28704000000000002e03,
    "ep": 0.62197183098591557e00,
    "ep1": 0.60778985507246353e00,
    "ep2": 0.16077898550724636e01,
    "Lv": 0.25000000000000000e07,
    "grav": 0.98100000000000005e01,
    "eps": 0.10000000000000000e-09,
}


def get_mixing_length(ds):
    nzm, nzt, ngrdcol, zm, zt, dzm, dzt, invrs_dzm, invrs_dzt = sam.get_grid(ds)

    thvm = sam.stagger_var("THETAV", ds, zt)
    thlm = sam.stagger_var("THETAL", ds, zt)
    rtm = sam.stagger_var("RTM", ds, zt)

    U2 = sam.stagger_var("U2", ds, zm)
    V2 = sam.stagger_var("V2", ds, zm)
    W2 = sam.stagger_var("W2", ds, zm)
    em = 0.5 * (U2 + V2 + W2)

    Lscale_max = (
        0.25 * 64 * 100
    )  # 64 pts * 100 m dx_LES = dx_GCM, CLUBB takes 1/4 this for max when implemented

    p_in_Pa = (
        sam.stagger_var("p", ds, "zt") * 100
    )  # p in mb * 10^-3 bar/mb * 10^5 Pa/bar

    cp = 1004.0  # Specific heat capacity of air at constant pressure
    Rd = 287.0  # Gas constant for dry air
    exner = (p_in_Pa / 10**5) ** (Rd / cp)

    thv_ds = sam.stagger_var("THETA", ds, "zt") * (1 + 0.608 * rtm)

    mu = np.full(ngrdcol, 1.0e-3)

    lmin = 20.0

    saturation_formula = 1
    l_implemented = True

    Lscale, Lscale_up, Lscale_down = compute_mixing_length(
        nzm,
        nzt,
        ngrdcol,
        zm,
        zt,
        dzm,
        dzt,
        invrs_dzm,
        invrs_dzt,
        thvm,
        thlm,
        rtm,
        em,
        Lscale_max,
        p_in_Pa,
        exner,
        thv_ds,
        mu,
        lmin,
        saturation_formula,
        l_implemented,
    )

    return Lscale, Lscale_up, Lscale_down


def compute_mixing_length(
    nzm,
    nzt,
    ngrdcol,
    zm,
    zt,
    dzm,
    dzt,
    invrs_dzm,
    invrs_dzt,
    thvm,
    thlm,
    rtm,
    em,
    Lscale_max,
    p_in_Pa,
    exner,
    thv_ds,
    mu,
    lmin,
    saturation_formula,
    l_implemented=True,
):
    """
    Compute the mixing length using Larson's 5th moist, nonlocal length scale method.
    """

    # Constants
    zlmin = 0.1
    Lscale_sfclyr_depth = 500.0
    grav = CLUBB_STANDALONE_CONSTANTS["grav"]
    cp = CLUBB_STANDALONE_CONSTANTS["Cp"]
    Rd = CLUBB_STANDALONE_CONSTANTS["Rd"]
    Lv = CLUBB_STANDALONE_CONSTANTS["Lv"]
    eps = CLUBB_STANDALONE_CONSTANTS["eps"]
    ep = CLUBB_STANDALONE_CONSTANTS["ep"]
    ep1 = CLUBB_STANDALONE_CONSTANTS["ep1"]
    ep2 = CLUBB_STANDALONE_CONSTANTS["ep2"]

    # Compute turbulent kinetic energy (TKE)
    tke_i = (em[:, :-1] + em[:, 1:]) / 2

    # Initialize output arrays
    Lscale = np.full((ngrdcol, nzt), zlmin)
    Lscale_up = np.full((ngrdcol, nzt), zlmin)
    Lscale_down = np.full((ngrdcol, nzt), zlmin)

    grav_on_thvm = grav / thvm
    Lv_coef = Lv / (exner * cp) - ep2 * thv_ds

    exp_mu_dzm = np.exp(-mu[:, None] * dzm)
    invrs_dzm_on_mu = invrs_dzm / mu[:, None]
    entrain_coef = (1 - exp_mu_dzm) * invrs_dzm_on_mu

    # Compute auxiliary values
    Lv2_coef = ep * Lv**2 / (Rd * cp)
    invrs_Lscale_sfclyr_depth = 1.0 / Lscale_sfclyr_depth

    ## Upward Mixing Length Calculation

    thl_par_j_precalc = np.full(thlm.shape, np.nan)
    thl_par_j_precalc[:, 1:-1] = (
        thlm[:, 1:-1]
        - thlm[:, 0:-2] * exp_mu_dzm[:, 1:-2]
        - (thlm[:, 1:-1] - thlm[:, 0:-2]) * entrain_coef[:, 1:-2]
    )

    rt_par_j_precalc = np.full(rtm.shape, np.nan)
    rt_par_j_precalc[:, 1:-1] = (
        rtm[:, 1:-1]
        - rtm[:, 0:-2] * exp_mu_dzm[:, 1:-2]
        - (rtm[:, 1:-1] - rtm[:, 0:-2]) * entrain_coef[:, 1:-2]
    )

    thl_par_1 = np.full(thlm.shape, np.nan)
    thl_par_1[:, 1:] = (
        thlm[:, 1:] - (thlm[:, 1:] - thlm[:, :-1]) * entrain_coef[:, 1:-1]
    )

    tl_par_1 = thl_par_1 * exner

    rt_par_1 = np.full(rtm.shape, np.nan)
    rt_par_1[:, 1:] = rtm[:, 1:] - (rtm[:, 1:] - rtm[:, :-1]) * entrain_coef[:, 1:-1]

    rsatl_par_1 = sat_mixrat_liq(
        tl_par_1, p_in_Pa, saturation_formula=saturation_formula
    )

    tl_par_1_sqd = tl_par_1**2

    s_par_1 = (
        (rt_par_1 - rsatl_par_1)
        * tl_par_1_sqd
        / (tl_par_1_sqd + Lv2_coef * rsatl_par_1)
    )

    rc_par_1 = np.maximum(s_par_1, 0.0)

    thv_par_1 = thl_par_1 + ep1 * thv_ds * rt_par_1 + Lv_coef * rc_par_1

    dCAPE_dz_1 = grav_on_thvm * (thv_par_1 - thvm)

    CAPE_incr_1 = 0.5 * dCAPE_dz_1 * dzm[:, 1:]

    for i in range(ngrdcol):
        Lscale_up_max_alt = 0.0  # Initial max height for upward mixing
        for k in range(nzt - 2):
            if tke_i[i, k] + CAPE_incr_1[i, k + 1] > 0.0:
                tke = tke_i[i, k] + CAPE_incr_1[i, k + 1]
                j = k + 2

                thl_par_j = thl_par_1[i, k + 1]
                rt_par_j = rt_par_1[i, k + 1]
                dCAPE_dz_j_minus_1 = dCAPE_dz_1[i, k + 1]

                while j < nzt - 1:
                    thl_par_j = thl_par_j_precalc[i, j] + thl_par_j * exp_mu_dzm[i, j]

                    rt_par_j = rt_par_j_precalc[i, j] + rt_par_j * exp_mu_dzm[i, j]

                    tl_par_j = thl_par_j * exner[i, j]

                    rsatl_par_j = sat_mixrat_liq(
                        tl_par_j, p_in_Pa[i, j], saturation_formula=saturation_formula
                    )

                    tl_par_j_sqd = tl_par_j**2

                    s_par_j = (
                        (rt_par_j - rsatl_par_j)
                        * tl_par_j_sqd
                        / (tl_par_j_sqd + Lv2_coef * rsatl_par_j)
                    )

                    rc_par_j = np.maximum(s_par_j, 0.0)

                    thv_par_j = (
                        thl_par_j
                        + ep1 * thv_ds[i, j] * rt_par_j
                        + Lv_coef[i, j] * rc_par_j
                    )

                    dCAPE_dz_j = grav_on_thvm[i, j] * (thv_par_j - thvm[i, j])

                    CAPE_incr = 0.5 * (dCAPE_dz_j + dCAPE_dz_j_minus_1) * dzm[i, j]

                    if tke + CAPE_incr <= 0:
                        break

                    dCAPE_dz_j_minus_1 = dCAPE_dz_j

                    tke += CAPE_incr

                    j += 1

                Lscale_up[i, k] += zt[i, j - 1] - zt[i, k]  #'Case 1.1'

                if j < nzt - 1:
                    if (
                        np.abs(dCAPE_dz_j - dCAPE_dz_j_minus_1) * 2
                        <= np.abs(dCAPE_dz_j + dCAPE_dz_j_minus_1) * eps
                    ):
                        Lscale_up[i, k] += -tke / dCAPE_dz_j
                        # print('Case 1.2')
                        # print(Lscale_up[i])
                        # print(Lscale_up[i,k])
                    else:
                        invrs_dCAPE_diff = 1.0 / (dCAPE_dz_j - dCAPE_dz_j_minus_1)

                        Lscale_up[i, k] += (
                            -dCAPE_dz_j_minus_1 * invrs_dCAPE_diff * dzm[i, j]
                            - np.sqrt(
                                dCAPE_dz_j_minus_1**2
                                - 2.0
                                * tke
                                * invrs_dzm[i, j]
                                * (dCAPE_dz_j - dCAPE_dz_j_minus_1)
                            )
                            * invrs_dCAPE_diff
                            * dzm[i, j]
                        )
                        # print('Case 1.3')
                        # print(Lscale_up[i])
                        # print(Lscale_up[i,k])

                # print('Case 1.1')
                # print(Lscale_up[i])
                # print(Lscale_up[i,k])

            else:
                Lscale_up[i, k] += (
                    -np.sqrt(-2.0 * tke_i[i, k] * dzm[i, k + 1] * dCAPE_dz_1[i, k + 1])
                    / dCAPE_dz_1[i, k + 1]
                )
                # print('Case 2')
                # print(Lscale_up[i])
                # print(Lscale_up[i,k])

            if zt[i, k] + Lscale_up[i, k] < Lscale_up_max_alt:
                Lscale_up[i, k] = Lscale_up_max_alt - zt[i, k]
                # print('Nonlocal')
                # print(Lscale_up[i])
            else:
                Lscale_up_max_alt = Lscale_up[i, k] + zt[i, k]
                # print('Local')
                # print(Lscale_up[i])
            # print('max now'+str(Lscale_up_max_alt))

    ## Downward Mixing Length Calculation

    thl_par_j_precalc[:, :-1] = (
        thlm[:, :-1]
        - thlm[:, 1:] * exp_mu_dzm[:, 1:-1]
        - (thlm[:, :-1] - thlm[:, 1:]) * entrain_coef[:, 1:-1]
    )

    rt_par_j_precalc[:, :-1] = (
        rtm[:, :-1]
        - rtm[:, 1:] * exp_mu_dzm[:, 1:-1]
        - (rtm[:, :-1] - rtm[:, 1:]) * entrain_coef[:, 1:-1]
    )

    thl_par_1[:, :-1] = (
        thlm[:, :-1] - (thlm[:, :-1] - thlm[:, 1:]) * entrain_coef[:, 1:-1]
    )

    tl_par_1 = thl_par_1 * exner

    rt_par_1[:, :-1] = rtm[:, :-1] - (rtm[:, :-1] - rtm[:, 1:]) * entrain_coef[:, 1:-1]

    rsatl_par_1 = sat_mixrat_liq(
        tl_par_1, p_in_Pa, saturation_formula=saturation_formula
    )

    tl_par_1_sqd = tl_par_1**2

    s_par_1 = (
        (rt_par_1 - rsatl_par_1)
        * tl_par_1_sqd
        / (tl_par_1_sqd + Lv2_coef * rsatl_par_1)
    )

    rc_par_1 = np.maximum(s_par_1, 0.0)

    thv_par_1 = thl_par_1 + ep1 * thv_ds * rt_par_1 + Lv_coef * rc_par_1

    dCAPE_dz_1 = grav_on_thvm * (thv_par_1 - thvm)

    CAPE_incr_1 = 0.5 * dCAPE_dz_1 * dzm[:, 1:]

    for i in range(ngrdcol):
        Lscale_down_min_alt = zt[i, -1]
        for k in range(nzt - 1, 0, -1):
            if tke_i[i, k] - CAPE_incr_1[i, k - 1] > 0.0:
                tke = tke_i[i, k] - CAPE_incr_1[i, k - 1]
                j = k - 2

                thl_par_j = thl_par_1[i, k - 1]
                rt_par_j = rt_par_1[i, k - 1]
                dCAPE_dz_j_plus_1 = dCAPE_dz_1[i, k - 1]

                while j >= 0:
                    thl_par_j = (
                        thl_par_j_precalc[i, j] + thl_par_j * exp_mu_dzm[i, j + 1]
                    )

                    rt_par_j = rt_par_j_precalc[i, j] + rt_par_j * exp_mu_dzm[i, j + 1]

                    tl_par_j = thl_par_j * exner[i, j]

                    rsatl_par_j = sat_mixrat_liq(
                        tl_par_j, p_in_Pa[i, j], saturation_formula=saturation_formula
                    )

                    tl_par_j_sqd = tl_par_j**2

                    s_par_j = (
                        (rt_par_j - rsatl_par_j)
                        * tl_par_j_sqd
                        / (tl_par_j_sqd + Lv2_coef * rsatl_par_j)
                    )

                    rc_par_j = np.maximum(s_par_j, 0.0)

                    thv_par_j = (
                        thl_par_j
                        + ep1 * thv_ds[i, j] * rt_par_j
                        + Lv_coef[i, j] * rc_par_j
                    )

                    dCAPE_dz_j = grav_on_thvm[i, j] * (thv_par_j - thvm[i, j])

                    CAPE_incr = 0.5 * (dCAPE_dz_j + dCAPE_dz_j_plus_1) * dzm[i, j + 1]

                    if tke - CAPE_incr <= 0:
                        break

                    dCAPE_dz_j_plus_1 = dCAPE_dz_j

                    tke -= CAPE_incr

                    j -= 1

                Lscale_down[i, k] += zt[i, k] - zt[i, j + 1]

                if j >= 0:
                    if (
                        np.abs(dCAPE_dz_j - dCAPE_dz_j_plus_1) * 2
                        <= np.abs(dCAPE_dz_j + dCAPE_dz_j_plus_1) * eps
                    ):
                        Lscale_down[i, k] += tke / dCAPE_dz_j
                    else:
                        invrs_dCAPE_diff = 1.0 / (dCAPE_dz_j - dCAPE_dz_j_plus_1)

                        Lscale_down[i, k] += (
                            -dCAPE_dz_j_plus_1 * invrs_dCAPE_diff * dzm[i, j + 1]
                            + np.sqrt(
                                dCAPE_dz_j_plus_1**2
                                + 2.0
                                * tke
                                * invrs_dzm[i, j + 1]
                                * (dCAPE_dz_j - dCAPE_dz_j_plus_1)
                            )
                            * invrs_dCAPE_diff
                            * dzm[i, j + 1]
                        )

            else:
                Lscale_down[i, k] += (
                    np.sqrt(2.0 * tke_i[i, k] * dzm[i, k] * dCAPE_dz_1[i, k - 1])
                    / dCAPE_dz_1[i, k - 1]
                )

            if zt[i, k] - Lscale_down[i, k] > Lscale_down_min_alt:
                Lscale_down[i, k] = zt[i, k] - Lscale_down_min_alt
            else:
                Lscale_down_min_alt = zt[i, k] - Lscale_down[i, k]

    ## Final Mixing Length Calculation

    if l_implemented:
        lminh = (
            np.maximum(0.0, Lscale_sfclyr_depth - (zt - zm[:, 0:1]))
            * lmin
            * invrs_Lscale_sfclyr_depth
        )
    else:
        lminh = (
            np.maximum(0.0, Lscale_sfclyr_depth - zt) * lmin * invrs_Lscale_sfclyr_depth
        )

    Lscale_up = np.maximum(lminh, Lscale_up)
    Lscale_down = np.maximum(lminh, Lscale_down)

    Lscale = np.sqrt(Lscale_up * Lscale_down)

    # Lscale[:, -1] = Lscale[:, -2]
    Lscale = np.minimum(Lscale, Lscale_max)
    return Lscale, Lscale_up, Lscale_down


class SaturationFormula(enum.Enum):
    """
    Mirrors avaliable saturation formula options in CLUBB.

    Taken from:
    https://github.com/m2lines/clubb_ML/blob/master/src/CLUBB_core/model_flags.F90#L120-L125
    """

    BOLTON = 1
    GFDL = 2
    FLATAU = 3
    LOOKUP = 4


def _sat_flatau(T):
    T_FREEZE_K = 273.15
    MIN_T_IN_C = -85.0

    T_in_C = T - T_FREEZE_K
    T_in_C = np.maximum(T_in_C, MIN_T_IN_C)

    T_in_C_sqd = T_in_C**2

    return (
        -3.21582393e-14
        * (T_in_C - 646.5835252598777)
        * (T_in_C + 90.72381630364440)
        * (T_in_C_sqd + 111.0976961559954 * T_in_C + 6459.629194243118)
        * (T_in_C_sqd + 152.3131930092453 * T_in_C + 6499.774954705265)
        * (T_in_C_sqd + 174.4279584934021 * T_in_C + 7721.679732114084)
    )


@typing.overload
def sat_mixrat_liq(
    T: FLOAT_T,
    p: FLOAT_T,
    saturation_formula: SaturationFormula | int = SaturationFormula.BOLTON,
) -> FLOAT_T: ...
@typing.overload
def sat_mixrat_liq(
    T: npt.NDArray[FLOAT_T],
    p: npt.NDArray[FLOAT_T],
    saturation_formula: SaturationFormula | int = SaturationFormula.BOLTON,
) -> npt.NDArray[FLOAT_T]: ...


def sat_mixrat_liq(T, p, saturation_formula=SaturationFormula.BOLTON):
    """
    Computes the saturation mixing ratio over liquid water.

    Based on the implementation in CLUBB `saturation` module. Avaiable at:

    https://github.com/m2lines/clubb_ML/blob/92b8d7aeeafc1b045641b4c91806144a1c68945b/src/CLUBB_core/saturation.F90

    Parameters
    ----------
    T : np.ndarray | float
        Temperature [K]
    p : np.ndarray | float
        Pressure [Pa]
    saturation_formula : SaturationFormula | int, optional
        The saturation formula to use. The default is SaturationFormula.BOLTON.

    Returns
    -------
    np.ndarray | float
        Saturation mixing ratio over liquid water.
    """
    saturation_formula = SaturationFormula(saturation_formula)
    # Ratio of molecular weights (water vapor/dry air)
    epsilon = CLUBB_STANDALONE_CONSTANTS["ep"]

    match saturation_formula:
        case SaturationFormula.BOLTON:
            es = (
                6.112 * np.exp((17.67 * (T - 273.15)) / (T - 29.65)) * 100
            )  # Convert hPa to Pa
        case SaturationFormula.FLATAU:
            es = _sat_flatau(T)
        case _:
            raise NotImplementedError(
                f"Saturation formula {saturation_formula} in not implemented yet."
            )
    return epsilon * es / (p - es)
