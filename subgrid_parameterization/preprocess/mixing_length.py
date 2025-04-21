import numpy as np
from subgrid_parameterization.preprocess import SAM_helpers as sam

def get_mixing_length(ds):
    
    nzm, nzt, ngrdcol, zm, zt, dzm, dzt, invrs_dzm, invrs_dzt = sam.get_grid(ds)
    
    thvm = sam.stagger_var('THETAV',ds,zt)
    thlm = sam.stagger_var('THETAL',ds,zt)
    rtm = sam.stagger_var('RTM',ds,zt)

    Lscale_max = 0.25*64*100 # 64 pts * 100 m dx_LES = dx_GCM, CLUBB takes 1/4 this for max when implemented

    U2 = sam.stagger_var('U2',ds,zm)
    V2 = sam.stagger_var('V2',ds,zm)
    W2 = sam.stagger_var('W2',ds,zm)
    em = 0.5*np.sqrt(U2 + V2 + W2)

    


def compute_mixing_length(nzm, nzt, ngrdcol, zm, zt, dzm, dzt, invrs_dzm, invrs_dzt, 
                          thvm, thlm, rtm, em, 
                          Lscale_max, p_in_Pa, exner, thv_ds, mu, lmin, 
                          saturation_formula, l_implemented=True):
    """
    Compute the mixing length using Larson's 5th moist, nonlocal length scale method.
    """
    
    # Constants
    zlmin = 0.1
    Lscale_sfclyr_depth = 500.0
    grav = 9.81  # gravitational acceleration
    cp = 1004.0  # Specific heat capacity of air at constant pressure
    Rd = 287.0  # Gas constant for dry air
    Rv = 461.0  # Gas constant for dry air
    Lv = 2.5e6   # Latent heat of vaporization
    eps = 1e-6   # Small threshold value
    ep = Rd / Rv
    ep1 = (1 - ep) / ep
    ep2 = 1 / ep
    
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
    
    thl_par_j_precalc = np.full(thlm.shape,np.nan)
    thl_par_j_precalc[:, 1:-1] = thlm[:, 1:-1] - thlm[:, 0:-2] * exp_mu_dzm[:, 1:-2] \
                             - ( thlm[:, 1:-1] - thlm[:, 0:-2] ) * entrain_coef[:, 1:-2]

    
    rt_par_j_precalc = np.full(rtm.shape,np.nan) 
    rt_par_j_precalc[:, 1:-1] = rtm[:, 1:-1] - rtm[:, 0:-2] * exp_mu_dzm[:, 1:-2] \
                            - ( rtm[:, 1:-1] - rtm[:, 0:-2]) * entrain_coef[:, 1:-2]
    
    thl_par_1 = np.full(thlm.shape,np.nan) 
    thl_par_1[:, 1:] = thlm[:, 1:] - (thlm[:, 1:] - thlm[:, :-1]) * entrain_coef[:, 1:-1]

    tl_par_1 = thl_par_1*exner
    
    rt_par_1 = np.full(rtm.shape,np.nan) 
    rt_par_1[:, 1:] = rtm[:, 1:] - (rtm[:, 1:] - rtm[:, :-1]) * entrain_coef[:, 1:-1]

    rsatl_par_1 = sat_mixrat_liq(tl_par_1, p_in_Pa)
    
    tl_par_1_sqd = tl_par_1**2
    
    s_par_1 = (rt_par_1 - rsatl_par_1) * tl_par_1_sqd / ( tl_par_1_sqd + Lv2_coef * rsatl_par_1 )

    rc_par_1 = np.maximum( s_par_1, 0.0 )

    thv_par_1 = thl_par_1 + ep1 * thv_ds * rt_par_1 + Lv_coef * rc_par_1

    dCAPE_dz_1 = grav_on_thvm * ( thv_par_1 - thvm )

    CAPE_incr_1 = 0.5 * dCAPE_dz_1 * dzm[:,1:]

    for i in range(ngrdcol):
        Lscale_up_max_alt = 0.0  # Initial max height for upward mixing
        for k in range(nzt - 2):

            if tke_i[i, k] + CAPE_incr_1[i, k + 1] > 0.0:
                tke = tke_i[i, k] + CAPE_incr_1[i, k + 1]
                j = k + 2
                
                thl_par_j = thl_par_1[i, k + 1]
                rt_par_j = rt_par_1[i, k + 1]
                dCAPE_dz_j_minus_1 = dCAPE_dz_1[i, k+1]
                
                while j < nzt-1:

                    thl_par_j = thl_par_j_precalc[i, j] + thl_par_j * exp_mu_dzm[i, j]
                    
                    rt_par_j = rt_par_j_precalc[i, j] + rt_par_j * exp_mu_dzm[i, j]

                    tl_par_j = thl_par_j*exner[i,j]

                    rsatl_par_j = sat_mixrat_liq(tl_par_j,  p_in_Pa[i,j])

                    tl_par_j_sqd = tl_par_j**2

                    s_par_j = ( rt_par_j - rsatl_par_j ) * tl_par_j_sqd \
                              / ( tl_par_j_sqd + Lv2_coef * rsatl_par_j )

                    rc_par_j = np.maximum( s_par_j, 0.0 )

                    thv_par_j = thl_par_j + ep1 * thv_ds[i, j] * rt_par_j + Lv_coef[i,j] * rc_par_j
                    
                    dCAPE_dz_j = grav_on_thvm[i, j] * (thv_par_j - thvm[i, j])

                    CAPE_incr = 0.5 * (dCAPE_dz_j + dCAPE_dz_j_minus_1) * dzm[i, j]

                    if tke + CAPE_incr <= 0:
                        break

                    dCAPE_dz_j_minus_1 = dCAPE_dz_j
                    
                    tke += CAPE_incr
                    
                    j += 1

                Lscale_up[i, k] += zt[i, j - 1] - zt[i, k] #'Case 1.1'

                if j < nzt-1:
                    if ( np.abs( dCAPE_dz_j - dCAPE_dz_j_minus_1 ) * 2 <= \
                         np.abs( dCAPE_dz_j + dCAPE_dz_j_minus_1 ) * eps ):
                        Lscale_up[i, k] += (-tke / dCAPE_dz_j)
                        # print('Case 1.2')
                        # print(Lscale_up[i])
                        # print(Lscale_up[i,k])
                    else:
                        invrs_dCAPE_diff = 1.0 / ( dCAPE_dz_j - dCAPE_dz_j_minus_1 )
                        
                        Lscale_up[i,k] += -dCAPE_dz_j_minus_1 * invrs_dCAPE_diff * dzm[i,j] \
                                         - np.sqrt( dCAPE_dz_j_minus_1**2 - 2.0 * tke * invrs_dzm[i,j] \
                                            * ( dCAPE_dz_j - dCAPE_dz_j_minus_1 ) ) \
                                         * invrs_dCAPE_diff  * dzm[i,j]
                        # print('Case 1.3')
                        # print(Lscale_up[i])
                        # print(Lscale_up[i,k])

                # print('Case 1.1')
                # print(Lscale_up[i])
                # print(Lscale_up[i,k])
            
            else:
                Lscale_up[i,k] += - np.sqrt( -2.0 * tke_i[i,k] * dzm[i,k+1] * dCAPE_dz_1[i,k+1]  ) \
                                    / dCAPE_dz_1[i,k+1]
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

    thl_par_j_precalc[:, :-1] = thlm[:, :-1] - thlm[:, 1:] * exp_mu_dzm[:, 1:-1] \
                            - ( thlm[:, :-1] - thlm[:, 1:]) * entrain_coef[:, 1:-1]
    
    rt_par_j_precalc[:, :-1] = rtm[:, :-1] - rtm[:, 1:] * exp_mu_dzm[:, 1:-1] \
                            - (rtm[:, :-1] - rtm[:, 1:]) * entrain_coef[:, 1:-1]

    
    thl_par_1[:, :-1] = thlm[:, :-1] - (thlm[:, :-1] - thlm[:, 1:]) * entrain_coef[:, 1:-1]

    tl_par_1 = thl_par_1*exner
    
    rt_par_1[:, :-1] = rtm[:, :-1] - (rtm[:, :-1] - rtm[:, 1:] ) * entrain_coef[:, 1:-1]

    rsatl_par_1 = sat_mixrat_liq(tl_par_1, p_in_Pa)

    tl_par_1_sqd = tl_par_1**2

    s_par_1 = (rt_par_1 - rsatl_par_1) * tl_par_1_sqd / ( tl_par_1_sqd + Lv2_coef * rsatl_par_1 )

    rc_par_1 = np.maximum( s_par_1, 0.0 )

    thv_par_1 = thl_par_1 + ep1 * thv_ds * rt_par_1 + Lv_coef * rc_par_1

    dCAPE_dz_1 = grav_on_thvm * ( thv_par_1 - thvm )

    CAPE_incr_1 = 0.5 * dCAPE_dz_1 * dzm[:,1:]
    
    for i in range(ngrdcol):
        Lscale_down_min_alt = zt[i, -1]
        for k in range(nzt - 1, 1, -1):
            if tke_i[i, k] - CAPE_incr_1[i, k - 1] > 0.0:
                tke = tke_i[i, k] - CAPE_incr_1[i, k - 1]
                j = k - 2
    
                thl_par_j = thl_par_1[i, k - 1]
                rt_par_j = rt_par_1[i, k - 1]
                dCAPE_dz_j_plus_1 = dCAPE_dz_1[i, k-1]

                while j >= 0:
  
                    thl_par_j = thl_par_j_precalc[i, j] + thl_par_j * exp_mu_dzm[i, j+1]
                    
                    rt_par_j = rt_par_j_precalc[i, j] + rt_par_j * exp_mu_dzm[i, j+1]

                    tl_par_j = thl_par_j*exner[i,j]

                    rsatl_par_j = sat_mixrat_liq(tl_par_j,  p_in_Pa[i,j])

                    tl_par_j_sqd = tl_par_j**2

                    s_par_j = ( rt_par_j - rsatl_par_j ) * tl_par_j_sqd \
                              / ( tl_par_j_sqd + Lv2_coef * rsatl_par_j )

                    rc_par_j = np.maximum( s_par_j, 0.0 )

                    thv_par_j = thl_par_j + ep1 * thv_ds[i, j] * rt_par_j + Lv_coef[i,j] * rc_par_j
                    
                    dCAPE_dz_j = grav_on_thvm[i, j] * (thv_par_j - thvm[i, j])

                    CAPE_incr = 0.5 * (dCAPE_dz_j + dCAPE_dz_j_plus_1) * dzm[i, j+1]

                    if tke - CAPE_incr <= 0:
                        break

                    dCAPE_dz_j_plus_1 = dCAPE_dz_j
                    
                    tke -= CAPE_incr
                    
                    j -= 1

                Lscale_down[i, k] += zt[i, k] - zt[i,j + 1]

                if j >= 0:
                    if (np.abs( dCAPE_dz_j - dCAPE_dz_j_plus_1 ) * 2 <= \
                        np.abs( dCAPE_dz_j + dCAPE_dz_j_plus_1 ) * eps ):
                        Lscale_down[i, k] += tke / dCAPE_dz_j
                    else:
                        invrs_dCAPE_diff = 1.0 / ( dCAPE_dz_j - dCAPE_dz_j_plus_1 )
                        
                        Lscale_down[i,k] += -dCAPE_dz_j_plus_1 * invrs_dCAPE_diff * dzm[i,j+1] \
                                         + np.sqrt( dCAPE_dz_j_minus_1**2 + 2.0 * tke * invrs_dzm[i,j+1] \
                                            * ( dCAPE_dz_j - dCAPE_dz_j_plus_1 ) ) \
                                         * invrs_dCAPE_diff  * dzm[i,j+1]                    

            else:
                Lscale_down[i,k] +=  np.sqrt( 2.0 * tke_i[i,k] * dzm[i,k] * dCAPE_dz_1[i,k-1]  ) \
                                    / dCAPE_dz_1[i,k-1]
                
            if zt[i,k] - Lscale_down[i,k] > Lscale_down_min_alt:
                Lscale_down[i,k] = zt[i,k] - Lscale_down_min_alt
            else:
                Lscale_down_min_alt = zt[i,k] - Lscale_down[i,k]
        
    ## Final Mixing Length Calculation
    
    if l_implemented:
        lminh = np.maximum(0.0, Lscale_sfclyr_depth - ( zt - zm[:,0:1] ) ) \
                  * lmin * invrs_Lscale_sfclyr_depth

    Lscale_up = np.maximum( lminh, Lscale_up)
    Lscale_down = np.maximum( lminh, Lscale_down)
    
    Lscale = np.sqrt( Lscale_up * Lscale_down )

    # Lscale[:, -1] = Lscale[:, -2]
    Lscale = np.minimum(Lscale, Lscale_max)

    return Lscale, Lscale_up, Lscale_down

def sat_mixrat_liq(T, p):
    """
    Calculate the saturation mixing ratio given temperature (K) and pressure (Pa).
    Using the Clausius-Clapeyron equation approximation.
    """
    es = 6.112 * np.exp((17.67 * (T - 273.15)) / (T - 29.65)) * 100  # Convert hPa to Pa
    epsilon = 0.622  # Ratio of molecular weights (water vapor/dry air)
    return epsilon * es / (p - es)