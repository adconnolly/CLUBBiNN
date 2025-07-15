import numpy as np
from scipy.interpolate import interp1d

def get_TKE(ds, z='zt'):

    if isinstance(z, str):
        _, _, _, _, zt, _, _, _, _ = sam.get_grid(ds)
    
    U2 = stagger_var('U2',ds,zt)
    V2 = stagger_var('V2',ds,zt)
    W2 = stagger_var('W2',ds,zt)
    return 0.5*np.sqrt(U2 + V2 + W2)

def get_disp(ds, z='zt'):

    if isinstance(z, str):
        _, _, _, _, zt, _, _, _, _ = sam.get_grid(ds)

    U2DFSN = stagger_var('U2DFSN',ds,zt)
    V2DFSN = stagger_var('V2DFSN',ds,zt)
    return 0.5 * ( U2DFSN + V2DFSN )  

def stagger_var(var_str, ds, z='zt'):
    if isinstance(z, str):
        if z=='zm':
            _, _, z, _, _, _, _, _ = stagger_grid(ds)
        else: # Default to zt
            _, _, _, z, _, _, _, _ = stagger_grid(ds)
    
    if var_str=='p':
        vals=expand(ds[var_str].squeeze().values, ds)
    else:
        vals=ds[var_str].squeeze().values

    if len(z.shape)==1:
        return interp1d(ds['z'].values,vals,axis=1,fill_value='extrapolate')(z)
    else:
        assert len(z.shape)==2
        assert (np.diff(z,axis=0)==0).all() # non-constant grid not implemented yet
        return interp1d(ds['z'].values,vals,axis=1,fill_value='extrapolate')(z[0])

def get_grid(ds,time_for_space=True):

    nzm, nzt, zmv, ztv, dzmv, dztv, invrs_dzmv, invrs_dztv = stagger_grid(ds)

    if time_for_space:    
        ngrdcol = len(ds['time'])
    else:
        ngrdcol = None
        return nzm, nzt, ngrdcol, zmv, ztv, dzmv, dztv, invrs_dzmv, invrs_dztv

    zm = expand(zmv, ngrdcol)
    zt = expand(ztv, ngrdcol)
    dzm = expand(dzmv, ngrdcol)
    dzt = expand(dztv, ngrdcol)
    
    return nzm, nzt, ngrdcol, zm, zt, dzm, dzt, 1.0/dzm, 1.0/dzt 

def stagger_grid(z):
    try:
        z = z['z'].values #in case xarray Dataset is passed
    except:
        z = z
    
    nzm = (len(z) + 1)//2 #Adding surface then leapfrogging the averages to stagger
    nzt = nzm - 1
    
    zm=np.concatenate(([0],0.5*(z[1:2*nzm-1:2]+z[2:2*nzm-1:2])))
    zt=(zm[:-1]+zm[1:])/2


    dzm = np.concatenate( ( [2.0*(zt[0]-zm[0])], np.diff(zt), [np.diff(zt)[-1]] ) )
                # gr%dzm(i,1) = 2.0_core_rknd * ( gr%zt(i,1) - gr%zm(i,1) )  
                # gr%dzm(i,nzm) = gr%dzm(i,nzm-1) from subroutine setup_grid_heights in grid_class.F90
    dzt = np.diff(zm)

    invrs_dzm = 1.0/dzm
    invrs_dzt = 1.0/dzt
    
    return nzm, nzt, zm, zt, dzm, dzt, invrs_dzm, invrs_dzt

def expand(u, ngrdcol):
    try:
        assert isinstance(ngrdcol, int)
        return np.array([u for i in range(ngrdcol)])
    except:
        ngrdcol = len(ngrdcol['time']) #in case xarray Dataset is passed
        return np.array([u for i in range(ngrdcol)])

    
    


    
    

    