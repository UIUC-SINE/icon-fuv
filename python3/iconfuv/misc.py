import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import line_aa
from scipy.signal import correlate, convolve2d
from skimage.transform import radon
import netCDF4
from itertools import product

def size_equalizer(x, ref_size, mode='center', val=0):
    """
    Crop or zero-pad a 2D array so that it has the size `ref_size`.
    Both cropping and zero-padding are done such that the symmetry of the
    input signal is preserved.
    Args:
        x (ndarray): array which will be cropped/zero-padded
        ref_size (list): list containing the desired size of the array [r1,r2]
        mode (str): ('center', 'topleft') where x should be placed when zero padding
    Returns:
        ndarray that is the cropper/zero-padded version of the input
    """
    if len(x.shape) == 2:
        if x.shape[0] > ref_size[0]:
            pad_left, pad_right = 0, 0
            crop_left = 0 if mode == 'topleft' else (x.shape[0] - ref_size[0] + 1) // 2
            crop_right = crop_left + ref_size[0]
        else:
            crop_left, crop_right = 0, x.shape[0]
            pad_right = ref_size[0] - x.shape[0] if mode == 'topleft' else (ref_size[0] - x.shape[0]) // 2
            pad_left = ref_size[0] - pad_right - x.shape[0]
        if x.shape[1] > ref_size[1]:
            pad_top, pad_bottom = 0, 0
            crop_top = 0 if mode == 'topleft' else (x.shape[1] - ref_size[1] + 1) // 2
            crop_bottom = crop_top + ref_size[1]
        else:
            crop_top, crop_bottom = 0, x.shape[1]
            pad_bottom = ref_size[1] - x.shape[1] if mode == 'topleft' else (ref_size[1] - x.shape[1]) // 2
            pad_top = ref_size[1] - pad_bottom - x.shape[1]

        # crop x
        cropped = x[crop_left:crop_right, crop_top:crop_bottom]
        # pad x
        padded = np.pad(
            cropped,
            ((pad_left, pad_right), (pad_top, pad_bottom)),
            mode='constant',
            constant_values=val
        )

        return padded

    elif len(x.shape) == 3:
        padded = np.zeros((x.shape[0], ref_size[0], ref_size[1]))
        for i in range(x.shape[0]):
            padded[i] = size_equalizer(x[i], ref_size)
        return padded

def liner(angle=np.arctan(15./8.), length=10):
    x1, y1 = length/2 * np.cos(angle), length/2 * np.sin(angle)
    x2, y2 = -x1, -y1
    c1, c2 = x1 + abs(x1), x2 + abs(x2)
    r1, r2 = -y1 + abs(y1), -y2 + abs(y2)
    r1, r2, c1, c2 = np.int(r1), np.int(r2), np.int(c1), np.int(c2)
    img = np.zeros((max(r1,r2)+1, max(c1,c2)+1))
    rr, cc, val = line_aa(r1, c1, r2, c2)
    img[rr, cc] = val
    # img = convolve2d(img, np.reshape(0.25*np.ones(4), (2,2)))
    return img

def orientation_finder_old(br):
    angle_list = np.linspace(0, np.pi, 180)
    angle_max = angle_list[0]
    energy_max = 0
    diff_ker = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])
    brd = correlate(br, diff_ker, mode='full')
    for angle in angle_list:
        feature = liner(angle=angle, length=20)
        feature_d = correlate(feature, diff_ker, mode='full')
        energy = np.linalg.norm(correlate(brd, feature_d, mode='full'))
        if energy > energy_max:
            angle_max = angle
            energy_max = energy
    return angle_max

def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product

def orientation_finder(br):
    rad = min(br.shape) // 2
    brf = np.fft.fftn(br)
    brf[0,:] = 0
    brf[:,0] = 0
    brfsa = abs(np.fft.fftshift(brf))
    theta = np.linspace(0., 180., 180, endpoint=False)
    sinogram = radon(brfsa, theta=theta, circle=True)
    sum = np.sum(sinogram[rad-10:rad+10], axis=0)
    return np.argmax(sum) * np.pi / 180.

def sliding_min(x, winsize=5, mode='reflect'):
    if len(x.shape) == 1:
        out = np.zeros_like(x)
        padfront = int((winsize-1)/2)
        padafter = winsize - 1 - padfront
        padded = np.pad(x, (padfront, padafter), mode=mode)
        for i in range(len(x)):
            out[i] = np.min(padded[i: i + winsize])
        return out
    elif len(x.shape) == 2:
        out = np.zeros_like(x)
        for i in range(x.shape[0]):
            out[i] = sliding_min(x[i])
        return out


def get_br(date='2020-01-03', epoch=300, stripe=None, v=2, r=0):
    file_l1='/home/kamo/resources/iconfuv/nc_files/ICON_L1_FUV_SWP_{}_v{:02d}r{:03d}.NC'.format(date,v,r)
    file_anc='/home/kamo/resources/iconfuv/nc_files/ICON_L0P_FUV_Ancillary_{}_v01r000.NC'.format(date)
    l1 = netCDF4.Dataset(file_l1, mode='r')
    anc = netCDF4.Dataset(file_anc, mode='r')
    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    mode = anc.variables['ICON_ANCILLARY_FUV_ACTIVITY'][:]
    mode_night = (mode == 258).astype(np.int)
    nights = np.diff(mode_night, prepend=0)
    nights[nights==-1] = 0
    idxs = np.where(mode==258)[0][:]
    nights = np.cumsum(nights)[idxs]
    orbits = anc.variables['ICON_ANCILLARY_FUV_ORBIT_NUMBER'][idxs]
    orbit = orbits[epoch]
    night = nights[epoch]
    orbit_ind = np.where(orbits==orbit)[0]
    night_ind = np.where(nights==night)[0]
    print('Epoch:{}\nOrbit:{}\nOrbit Inds:[{}-{}]'.format(epoch, orbit, orbit_ind[0], orbit_ind[-1]))
    print('Epoch:{}\nNight:{}\nNight Inds:[{}-{}]'.format(epoch, night, night_ind[0], night_ind[-1]))
    if stripe is not None:
        br = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[stripe]][idxs[night_ind],100:]
        br_err = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[stripe]][idxs[night_ind],100:].filled(fill_value=0)
        br.fill_value = br.min()
        br = br.filled()
        # try:
        #     br = np.array(br[br.mask==False].reshape((br.shape[0], -1)))
        # except:
        #     br = np.array()
        # l1.close()
        return br, br_err
    br = np.zeros((6, len(night_ind), 156))
    br_err = np.zeros((6, len(night_ind), 156))
    for i in range(6):
        br[i] = l1.variables['ICON_L1_FUVA_SWP_PROF_%s' % mirror_dir[i]][idxs[night_ind],100:].filled(fill_value=0)
        br_err[i] = l1.variables['ICON_L1_FUVA_SWP_PROF_%s_Error' % mirror_dir[i]][idxs[night_ind],100:].filled(fill_value=0)
    l1.close()
    return br, br_err


def get_br_nights(l1):
    mirror_dir = ['M9','M6','M3','P0','P3','P6']
    mode = l1.variables['ICON_L1_FUV_Mode'][:]
    mode_night = (mode == 2).astype(np.int)
    nights = np.diff(mode_night, prepend=0)
    nights[nights==-1] = 0
    idxs = np.where(mode==2)[0][:]
    nights = np.cumsum(nights)[idxs]
    brs = []
    brs_err = []
    mask_arr = []
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
        brs.append(br)
        brs_err.append(br_err)
        mask_arr.append(mask)
    return brs, brs_err, mask_arr, nights, idxs


def shiftappend(im, w=[5,5]):
    ss, aa, bb = im.shape
    r, c = w
    im2 = np.zeros((r*c*ss, aa+r-1, bb+c-1))
    im_large = size_equalizer(im, [aa+r-1, bb+c-1])
    sr = np.arange(r) - int((r-1)/2)
    sc = np.arange(c) - int((c-1)/2)
    sh = list(product(sr,sc))
    for i in range(r*c):
        im3 = np.roll(im_large, sh[i][0], axis=1)
        im2[ss*i: ss*(i+1)] = np.roll(im3, sh[i][1], axis=2)
    return size_equalizer(im2, [aa,bb])


def boxer(pt):
    pt2 = np.array([
        list(
            product(
                [pt[i, 0] - 1, pt[i, 0]], #pt[i, 0] + 1],
                [pt[i, 1] - 1, pt[i, 1]] #, pt[i, 1] + 1]
            )
        ) for i in range(pt.shape[0])
    ])
    return pt2.reshape((-1,2))


def labeler(br):
    br_masked = br.copy()
    mask = np.zeros_like(br)
    plt.figure(figsize=(8,8))
    plt.imshow(br_masked, vmax=300)
    plt.tight_layout()
    while True:
        pt = np.rint(np.asarray(plt.ginput(10, timeout=0))).astype(np.int)
        if pt.size is not 0:
            # pt = boxer(pt)
            br_masked[pt[:,1], pt[:,0]] = np.max(br_masked)
            mask[pt[:,1], pt[:,0]] = 1
        plt.imshow(br_masked, vmax=300)
        if plt.waitforbuttonpress():
            break
    return br_masked, mask

def labeler_init(br):
    br_masked = br.copy()
    plt.figure(figsize=(8,8))
    plt.imshow(br_masked, vmax=min(500, br.max()))
    plt.tight_layout()
    while True:
        pt = np.rint(np.asarray(plt.ginput(-1, timeout=1000))).astype(np.int)
        if len(pt) == 0:
            break
        if plt.waitforbuttonpress():
            break
    pt[:, 0] -= pt[:, 0].min()
    pt[:, 1] -= pt[:, 1].min()
    pt = np.flip(pt, axis=1)
    return pt

def feature_labeler(br, feature, mask=None, num=None, box=False):
    br_masked = br.copy()
    if box is True:
        feature = np.array([[0,0],[0,1],[1,0],[1,1]])
    if mask is not None:
        br_masked[mask==1] = min(np.max(br_masked), 500)
    else:
        mask = np.zeros_like(br)
    left = True
    if np.sum(feature, axis=1).min() > 0:
        left = False
    plt.figure(figsize=(8,8))
    plt.imshow(br_masked, vmax=min(500, br.max()))
    plt.tight_layout()
    while True:
        pt = np.rint(np.asarray(plt.ginput(2, timeout=1e3))).astype(np.int)
        plt.cla()
        plt.imshow(br_masked, vmax=min(500, br.max()))
        plt.show()
        if len(pt) == 0:
            break
        if left is False:
            pt[:, 0] -= feature[:, 1].max()
        pt = np.flip(pt, axis=1)
        ind = []
        for i in range(len(pt)):
            ind.append(feature + pt[i])
        ind = np.array(ind).reshape((-1,2))
        for i in ind:
            if (i[0] < br_masked.shape[0]) and (i[1] < br_masked.shape[1]):
                br_masked[i[0], i[1]] = np.max(br_masked)
                mask[i[0], i[1]] = 1
    # if num is not None:
    #     np.save('destarring/dataset/mask{}.npy'.format(num), mask)
    #     print('succesfully wrote mask{}.npy'.format(num))
    #     return 0
    # else:
    #     if len(os.listdir('./destarring/dataset')) == 0:
    #         num = 1
    #     else:
    #         num = 1
    #         for f in os.listdir('./destarring/dataset'):
    #             if f[0] == 'm':
    #                 fnum = int(f.split('.')[0].split('k')[-1])
    #                 num = fnum + 1 if fnum > num else num
    np.save('destarring/dataset/mask{}.npy'.format(num), mask)
