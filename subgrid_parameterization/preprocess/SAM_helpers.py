import numpy as np
from scipy.interpolate import interp1d

def stagger_grid(z):
    try:
        z = z['z'].values #in case an xarray Dataset is passed
    except:
        z = z
    
    nzm = (len(z) + 1)//2 #Adding surface then leapfrogging the averages to stagger
    nzt = nzm - 1
    
    zm=np.concatenate(([0],0.5*(z[1::2]+z[2::2])))
    zt=(zm[:-1]+zm[1:])/2

    dzm = np.concatenate(([np.nan],np.diff(zt)))
    dzt = np.diff(zm)
    invrs_dzm = 1.0/dzm
    invrs_dzt = 1.0/dzt

    
    return nzm, nzt, zm, zt, dzm, dzt, invrs_dzm, invrs_dzt

def stagger_var(var_str, ds, z='zt'):
    if isinstance(z, str):
        if z=='zm':
            _, _, z, _, _, _, _, _ = stagger_grid(ds)
        else: # Default to zt
            _, _, _, z, _, _, _, _ = stagger_grid(ds)
    
    if var_str=='p':
        zaxis=0
    else:
        zaxis=1
    
    return interp1d(ds['z'].values,ds[var_str].squeeze().values,axis=zaxis,fill_value='extrapolate')(z)
    

    