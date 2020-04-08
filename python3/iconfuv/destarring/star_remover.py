import numpy as np
from iconfuv.misc import get_br_nights
from airglow.FUV_L2 import l1_correction_orbit
import netCDF4

def star_removal(l1):
    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    mode = l1.variables['ICON_L1_FUV_Mode'][:]
    mode_night = (mode == 2).astype(np.int)
    nights = np.diff(mode_night, prepend=0)
    nights[nights==-1] = 0
    idxs = np.where(mode==2)[0][:]
    nights = np.cumsum(nights)[idxs]
    for night in np.unique(nights):
        night_ind = np.where(nights==night)[0]
        br = np.zeros((6, len(night_ind), 256))
        br_err = np.zeros((6, len(night_ind), 256))
        mask = np.zeros_like(br, dtype=np.bool)
        for i in range(6):
            tmp = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[i]][idxs[night_ind],:]
            br_err[i] = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[i]][idxs[night_ind],:].filled(fill_value=0)
            mask[i] = tmp.mask
            br[i] = tmp.filled(fill_value=0)
        br_corrected, br_err_modified = l1_correction_orbit(br, br_err)
        for i in range(6):
            l1.variables[
                'ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[i]
            ][idxs[night_ind],:] = np.ma.array(
                br_corrected[i], mask=mask[i]
            ).filled(fill_value=np.nan)
            l1.variables[
                'ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[i]
            ][idxs[night_ind],:] = np.ma.array(
                br_err_modified[i], mask=mask[i]
            ).filled(fill_value=np.nan)
    return l1
