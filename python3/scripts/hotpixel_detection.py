import numpy as np
import matplotlib.pyplot as plt
from iconfuv.destarring.methods import medfilt3d
from iconfuv.misc import get_br, sliding_min

# %% br
br, br_err = get_br(date='2020-01-13', epoch=250, v=3, r=0)
mask_med, br_medint, _ = medfilt3d(br, br_err)

projx = lambda br : np.mean(br, axis=1)
projy = lambda br : np.mean(br, axis=2)

# FIXME!! why are we operating on br instead of br_medint?
brx = projx(br_medint)
brx_min = sliding_min(brx)
br_cor = br_medint.copy()
for i in range(6):
    diff = brx[i] - brx_min[i]
    br_cor[i] -= diff
# br_cor[br_cor < 0] = br_medint[br_cor < 0]
br_cor[br_cor < 0] = 0
br_diff = br_medint - br_cor

mask_med2, br_medint2, _ = medfilt3d(br_cor, br_err, threshold=10)

# %% plotting
# for stripe in range(6):
#     fig, a = plt.subplots(2,2, figsize=(15, 7), gridspec_kw={'width_ratios': [3, 1]})
#     a[0,0].imshow(br_medint[stripe], aspect='auto'); a[0,0].set_title('Stripe: {}'.format(stripe))
#     a[0,0].get_shared_x_axes().join(a[0,0], a[1,0])
#     a[0,1].plot(projy(br_med)[stripe], range(len(projy(br_med)[stripe])))
#     a[0,1].invert_yaxis(); a[0,1].set_title('Projected Time Profile')
#     a[1,0].plot(brs[stripe], label='Average'); a[1,0].set_title('Projected Pixel Profile')
#     a[1,0].plot(brs_min[stripe], 'r', label='Moving Minimum on Average')
#     a[1,0].legend()
#     plt.tight_layout()
#     plt.show()

for stripe in range(6):
    fig, a = plt.subplots(4,2, figsize=(15, 8), gridspec_kw={'width_ratios': [3, 1]})
    a[0,0].get_shared_x_axes().join(a[0,0], a[1,0], a[2,0], a[3,0])
    a[0,1].get_shared_x_axes().join(a[0,1], a[1,1], a[2,1])
    a[0,0].imshow(br_medint[stripe], aspect='auto'); a[0,0].set_title('Input - Stripe: {}'.format(stripe))
    a[0,1].plot(projy(br_medint)[stripe], range(len(projy(br_medint)[stripe])))
    a[0,1].invert_yaxis(); a[0,1].set_title('Projected Time Profile')
    a[1,0].imshow(br_cor[stripe], aspect='auto'); a[1,0].set_title('Corrected - Stripe: {}'.format(stripe))
    a[1,1].plot(projy(br_cor)[stripe], range(len(projy(br_cor)[stripe])))
    a[1,1].invert_yaxis(); a[1,1].set_title('Projected Time Profile')
    a[2,0].imshow(br_medint2[stripe], aspect='auto'); a[2,0].set_title('Star Removed - Stripe: {}'.format(stripe))
    a[2,1].plot(projy(br_medint2)[stripe], range(len(projy(br_medint2)[stripe])))
    a[2,1].invert_yaxis(); a[2,1].set_title('Projected Time Profile')
    a[3,0].plot(projx(br_medint)[stripe], label='Input')
    a[3,0].plot(projx(br_cor)[stripe], label='Corrected')
    a[3,0].plot(projx(br_medint2)[stripe], label='Star Removed')
    a[3,0].legend()
    plt.tight_layout()
    plt.show()
    # plt.savefig('../../pres/pres7_2020-03-26_Hot_Pixel_Removal/result{}_v2.png'.format(stripe), transparent=True)
    # plt.close()
#
# fig, a = plt.subplots(2,1, figsize=(15, 7), sharex=True)
# a[0].imshow(np.mean(br_medint, axis=0), aspect='auto'); a[0].set_title('Average')
# a[1].plot(np.mean(brx, axis=0)); a[1].set_title('Projection')
# plt.tight_layout()
# plt.show()
