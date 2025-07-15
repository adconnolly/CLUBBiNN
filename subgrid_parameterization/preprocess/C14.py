import numpy as np
from subgrid_parameterization.preprocess import SAM_helpers as sam
from subgrid_parameterization.preprocess import mixing_length

def get_C14(ds):

    # nzm, nzt, ngrdcol, zm, zt, dzm, dzt, invrs_dzm, invrs_dzt
    _, _, _, _, zt, _, _, _, _ = sam.get_grid(ds)
    L, _, _ = mixing_length.get_mixing_length(ds)
    
    U2 = sam.stagger_var('U2',ds,zt)
    V2 = sam.stagger_var('V2',ds,zt)
    W2 = sam.stagger_var('W2',ds,zt)
    e = 0.5*np.sqrt(U2 + V2 + W2)

    U2DFSN = sam.stagger_var('U2DFSN',ds,zt)
    V2DFSN = sam.stagger_var('V2DFSN',ds,zt)
    disp = 0.5 * ( U2DFSN + V2DFSN ) # 
    
    return -3.0/2.0 * L / e**1.5 * disp

# def get_C14s(ds):

#     # nzm, nzt, ngrdcol, zm, zt, dzm, dzt, invrs_dzm, invrs_dzt
#     _, _, _, _, zt, _, _, _, _ = sam.get_grid(ds)
#     L, _, _ = mixing_length.get_mixing_length(ds)
    
#     U2 = sam.stagger_var('U2',ds,zt)
#     V2 = sam.stagger_var('V2',ds,zt)
#     W2 = sam.stagger_var('W2',ds,zt)
#     e = 0.5*np.sqrt(U2 + V2 + W2)

#     U2DFSN = sam.stagger_var('U2DFSN',ds,zt)
#     V2DFSN = sam.stagger_var('V2DFSN',ds,zt)
#     disp = 0.5 * ( U2DFSN + V2DFSN ) # 
    
#     return -3.0/2.0 * L / e**1.5 * disp