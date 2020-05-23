import numpy as np
import matplotlib.pyplot as plt
from airglow.inversion_helpers import custom_l25

date = '2020-01-20'
br_custom = True
method='derivative'
weight_resid = False
reg_order = 2
stripe = 3
epoch = 58

# %% inversion
(br,br_err,ver,ver_err,ne,ne_err,residual,seminorm,reg_corner,nes,
    vers,br_orig,br_err_orig,h,A)= custom_l25(date=date, method=method,
    epoch=epoch, stripe=stripe, l1_rev='v03r000', anc_rev='v01r001',
    weight_resid=weight_resid, reg_order=reg_order, iri_comp=False)

(br_iri,br_err_iri,ver_iri,ver_err_iri,ne_iri,ne_err_iri,residual,seminorm,reg_corner,nes,
    vers,br_orig,br_err_orig,h,A)= custom_l25(date=date, method=method,
    epoch=epoch, stripe=stripe, l1_rev='v03r000', anc_rev='v01r001',
    weight_resid=weight_resid, reg_order=reg_order, iri_comp=True)

# %% plotting
fig, ax = plt.subplots(ncols=2, figsize=(14,4.8))
ax[0].plot(br, h, label='Artifact Removed')
ax[0].plot(br_orig, h, label='Original')
ax[0].set_xlabel('Brightness [R]')
ax[0].set_ylabel('Tangent Altitude [km]')
ax[0].legend()
ax[0].set_title('Brightness Profile')
ax[0].ticklabel_format(scilimits=(0,3))

ax[1].plot(br_err, h, label='Artifact Removed')
ax[1].plot(br_err_orig, h, label='Original')
ax[1].set_xlabel('Brightness Error [R]')
ax[1].set_ylabel('Tangent Altitude [km]')
ax[1].legend()
ax[1].set_title('Brightness Error Profile')
ax[1].ticklabel_format(scilimits=(0,3))

# params = np.logspace(5.5,1,100)
params = np.logspace(5.5,1,100)
reg_corner_der = reg_corner
reg_der_ind = np.argmin(abs(params-reg_corner_der))
# reg_corner_cur = Maximum_Curvature_gradiens(residual,seminorm,params,method='curvature')
# param_list
for i in [0,25,50,75,-1]:
    reg_ratio = params[i] / reg_corner
    fig, ax = plt.subplots(ncols=3, figsize=(14,4.8))
    ax[0].plot(ver, h, label='Chosen VER')
    ax[0].plot(vers[i], h, label='Alternative VER')
    ax[0].set_xlabel('Volume Emission Rate [$ph/cm^3/s$]')
    ax[0].set_ylabel('Tangent Altitude [km]')
    ax[0].set_title('VER - Reg_Ratio:{}'.format(reg_ratio))
    ax[0].legend()
    ax[0].grid(which='both', axis='both')
    ax[0].ticklabel_format(scilimits=(0,3))

    ax[1].plot(ne, h, label='Chosen Ne')
    ax[1].plot(nes[i], h, label='Alternative Ne')
    ax[1].set_xlabel('O plus Density [$cm^{-3}$]')
    ax[1].set_ylabel('Tangent Altitude [km]')
    ax[1].set_title('Ne - Reg_Ratio:{}'.format(reg_ratio))
    ax[1].legend()
    ax[1].grid(which='both', axis='both')
    ax[1].ticklabel_format(scilimits=(0,3))

    ax[2].scatter(np.log(residual), np.log(seminorm), color='m')
    ax[2].scatter(np.log(residual[i]), np.log(seminorm[i]), color='orange', label='Alternative')
    ax[2].scatter(np.log(residual[reg_der_ind]), np.log(seminorm[reg_der_ind]), color='b', label='Chosen')
    ax[2].legend()
    ax[2].set_xlabel('Residual Norm: $||Ax-y||_2^2$')
    ax[2].set_ylabel('Regularizer Norm: $||Dx||_2^2$')
    ax[2].set_title('L-Curve')
    ax[2].grid(which='both', axis='both')
    ax[2].ticklabel_format(scilimits=(0,3))

    plt.tight_layout()
plt.show()
