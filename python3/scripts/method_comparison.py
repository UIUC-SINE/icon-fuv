import numpy as np
import matplotlib.pyplot as plt
import netCDF4, torch, glob
from iconfuv.destarring.methods import outlier3d, mf_filt, ft, medfilt3d, star_masker_orbit
from iconfuv.misc import get_br

# %% br
br_list = []
br_list.append(get_br(date='2020-01-03', epoch=1100, v=2, r=0))
br_list.append(get_br(date='2020-01-10', epoch=100, v=3, r=0))
# br_list.append(get_br(date='2020-01-10', epoch=100, v=2, r=1))
br_list.append(get_br(date='2020-01-11', epoch=100, v=3, r=0))
br_list.append(get_br(date='2020-01-12', epoch=100, v=3, r=0))
br_list.append(get_br(date='2020-01-13', epoch=100, v=3, r=0))
br_list.append(get_br(date='2020-01-14', epoch=100, v=3, r=0))
br_list.append(get_br(date='2020-01-15', epoch=100, v=3, r=0))

br_err_list = [i[1] for i in br_list]
br_list = [i[0] for i in br_list]

# %% removal
brf_list = []
br_med_list = []
br_medint_list = []
mask_unet_list = []
mask_mf_list = []
mask_out_list = []
mask_med_list = []

for br, br_err in zip(br_list,br_err_list):
    mask_unet = np.zeros_like(br)
    brf = np.zeros_like(br)
    mask_mf = np.zeros_like(br)
    # br_med, br_medint, mask_med = medfilt3d(br)
    mask_med, br_medint, _ = medfilt3d(br, br_err)
    for stripe in range(6):
        brf[stripe] = ft(br[stripe])
        mask_mf[stripe] = mf_filt(brf[stripe])
        mask_unet[stripe] = star_masker_orbit(br[stripe], stride=64, patch_size=64)
    brf_list.append(brf)
    mask_unet_list.append(mask_unet)
    # mask_out_list.append(outlier3d(br)[1])
    mask_mf_list.append(mask_mf)
    mask_med_list.append(mask_med)
    # br_med_list.append(br_med)
    br_medint_list.append(br_medint)

# %% plotting
num = 1
for stripe in range(6):
    fig, a = plt.subplots(1,4, figsize=(15.5, 4.8))
    a[0].imshow(br_list[num][stripe]); a[0].set_title('Stripe: {}'.format(stripe))
    # a[1].imshow(br_medint_list[num][stripe]); a[1].set_title('Median Interpolated')
    # a[2].imshow(br_med_list[num][stripe]); a[2].set_title('Median Med')
    a[1].imshow(mask_med_list[num][stripe]); a[1].set_title('3d Median Outlier Method')
    a[2].imshow(mask_mf_list[num][stripe]); a[2].set_title('Matched Filtered')
    # a[3].imshow(mask_out_list[num][stripe]) ; a[3].set_title('Outlier 3D')
    a[3].imshow(mask_unet_list[num][stripe]) ; a[3].set_title('UNet')
    plt.show()
    # plt.savefig('../../pres/pres6/methodcomp{}.png'.format(stripe), transparent=True)
    # plt.close()
