import numpy as np

def compute_mixing_length(nzm, nzt, ngrdcol, zm, zt, dzm, dzt, invrs_dzm, invrs_dzt, 
                          thvm, thlm, rtm, em, 
                          Lscale_max, p_in_Pa, exner, thv_ds, mu, lmin, 
                          saturation_formula, l_implemented):
    """
    Compute the mixing length using Larson's 5th moist, nonlocal length scale method.
    """
    
       
    # Constants
    zlmin = 0.1
    Lscale_sfclyr_depth = 500.0
    grav = 9.81  # gravitational acceleration
    Cp = 1004.0  # Specific heat capacity of air at constant pressure
    Rd = 287.0  # Gas constant for dry air
    Lv = 2.5e6   # Latent heat of vaporization
    eps = 1e-6   # Small threshold value
    ep = Rd / (Lv / Cp)
    ep1 = (1 - ep) / ep
    ep2 = 1 / ep

    # Initialize output arrays
    Lscale = np.full((ngrdcol, nzt), zlmin)
    Lscale_up = np.full((ngrdcol, nzt), zlmin)
    Lscale_down = np.full((ngrdcol, nzt), zlmin)

    # Initialize temporary arrays
    grav_on_thvm = grav / thvm
    Lv_coef = Lv / (exner * Cp) - ep2 * thv_ds

    exp_mu_dzm = np.exp(-mu[:, None] * dzm)
    print(mu.shape)
    print(mu[:, None].shape)
    print(invrs_dzm[:, None].shape)
    print(thlm.shape)
    invrs_dzm_on_mu = invrs_dzm / mu[:, None]
    print(invrs_dzm_on_mu[:, None].shape)
    entrain_coef = (1 - exp_mu_dzm) * invrs_dzm_on_mu

    # Compute turbulent kinetic energy (TKE)
    tke_i = zm2zt(em, ngrdcol)

    # Upward Mixing Length Calculation
    thl_par_j_precalc = thlm[:, 1:] - thlm[:, :-1] * exp_mu_dzm[:, 1:] - \
                        (thlm[:, 1:] - thlm[:, :-1]) * entrain_coef[:, 1:]

    rt_par_j_precalc = rtm[:, 1:] - rtm[:, :-1] * exp_mu_dzm[:, 1:] - \
                       (rtm[:, 1:] - rtm[:, :-1]) * entrain_coef[:, 1:]

    for i in range(ngrdcol):
        Lscale_up_max_alt = 0.0  # Initial max height for upward mixing
        for k in range(nzt - 2):
            if tke_i[i, k] > 0:
                tke = tke_i[i, k]
                j = k + 2
                thl_par_j = thl_par_j_precalc[i, k + 1]
                rt_par_j = rt_par_j_precalc[i, k + 1]
                dCAPE_dz_j_minus_1 = grav_on_thvm[i, k + 1] * (thl_par_j - thvm[i, k + 1])

                while j < nzt:
                    thl_par_j = thl_par_j_precalc[i, j] + thl_par_j * exp_mu_dzm[i, j]
                    rt_par_j = rt_par_j_precalc[i, j] + rt_par_j * exp_mu_dzm[i, j]

                    thv_par_j = thl_par_j + ep1 * thv_ds[i, j] * rt_par_j
                    dCAPE_dz_j = grav_on_thvm[i, j] * (thv_par_j - thvm[i, j])

                    CAPE_incr = 0.5 * (dCAPE_dz_j + dCAPE_dz_j_minus_1) * dzm[i, j]

                    if tke + CAPE_incr <= 0:
                        break

                    dCAPE_dz_j_minus_1 = dCAPE_dz_j
                    tke += CAPE_incr
                    j += 1

                Lscale_up[i, k] += zt[j - 1] - zt[k]

                if j < nzt:
                    Lscale_up[i, k] += (-tke / dCAPE_dz_j) if dCAPE_dz_j != 0 else 0

                if zt[k] + Lscale_up[i, k] < Lscale_up_max_alt:
                    Lscale_up[i, k] = Lscale_up_max_alt - zt[k]
                else:
                    Lscale_up_max_alt = zt[k] + Lscale_up[i, k]

    # Downward Mixing Length Calculation
    Lscale_down_min_alt = zt[-1]

    for i in range(ngrdcol):
        for k in range(nzt - 2, 1, -1):
            if tke_i[i, k] > 0:
                tke = tke_i[i, k]
                j = k - 2
                thl_par_j = thl_par_j_precalc[i, k - 1]
                rt_par_j = rt_par_j_precalc[i, k - 1]
                dCAPE_dz_j_plus_1 = grav_on_thvm[i, k - 1] * (thl_par_j - thvm[i, k - 1])

                while j >= 0:
                    thl_par_j = thl_par_j_precalc[i, j] + thl_par_j * exp_mu_dzm[i, j + 1]
                    rt_par_j = rt_par_j_precalc[i, j] + rt_par_j * exp_mu_dzm[i, j + 1]

                    thv_par_j = thl_par_j + ep1 * thv_ds[i, j] * rt_par_j
                    dCAPE_dz_j = grav_on_thvm[i, j] * (thv_par_j - thvm[i, j])

                    CAPE_incr = 0.5 * (dCAPE_dz_j + dCAPE_dz_j_plus_1) * dzm[j + 1]

                    if tke - CAPE_incr <= 0:
                        break

                    dCAPE_dz_j_plus_1 = dCAPE_dz_j
                    tke -= CAPE_incr
                    j -= 1

                Lscale_down[i, k] += zt[k] - zt[j + 1]

                if j >= 0:
                    Lscale_down[i, k] += (tke / dCAPE_dz_j) if dCAPE_dz_j != 0 else 0

                if zt[k] - Lscale_down[i, k] > Lscale_down_min_alt:
                    Lscale_down[i, k] = zt[k] - Lscale_down_min_alt
                else:
                    Lscale_down_min_alt = zt[k] - Lscale_down[i, k]

    # Final Mixing Length Calculation
    for i in range(ngrdcol):
        for k in range(nzt):
            lminh = max(0, Lscale_sfclyr_depth - zt[k]) * lmin / Lscale_sfclyr_depth
            Lscale_up[i, k] = max(lminh, Lscale_up[i, k])
            Lscale_down[i, k] = max(lminh, Lscale_down[i, k])
            Lscale[i, k] = np.sqrt(Lscale_up[i, k] * Lscale_down[i, k])

        Lscale[i, -1] = Lscale[i, -2]
        Lscale[i, :] = np.minimum(Lscale[i, :], Lscale_max[i])

    return Lscale, Lscale_up, Lscale_down
