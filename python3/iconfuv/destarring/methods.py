import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, convolve2d
from scipy.ndimage import median_filter
from iconfuv.misc import size_equalizer, liner, orientation_finder, get_br, shiftappend, correlation_coefficient
from iconfuv.destarring.nn_models import unet_v1
from iconfuv.data import unet_v1_path
import cv2, torch

def star_masker_orbit(br, patch_size=64, stride=32, nn_model=unet_v1, modelpath=unet_v1_path):
    # initialize the architecture
    model = nn_model()
    # read the parameters from the trained network
    model.load_state_dict(torch.load(modelpath))
    model = model.float()
    br = br.copy()
    h, w = br.shape
    counter = np.zeros_like(br)
    mask = np.zeros_like(br)
    row_list = list(np.arange(0, h-patch_size+1, stride))
    col_list = list(np.arange(0, w-patch_size+1, stride))
    row_list.append(h-patch_size)
    col_list.append(w-patch_size)
    for i in row_list:
        for j in col_list:
            patch = br[i:i+patch_size, j:j+patch_size]
            patch -= patch.mean()
            patch /= patch.std()
            mask[i:i+patch_size, j:j+patch_size] += np.array(
                model(
                    torch.from_numpy(
                        patch[np.newaxis, np.newaxis]
                    ).float()
                ).detach()[0,0,:,:]
            )
            counter[i:i+patch_size, j:j+patch_size] += 1
    mask /= counter
    mask = (mask > 0.5)
    return mask

def ft(br):
    rows, cols = br.shape
    winsize = 81
    brf = np.fft.fftn(br)
    brfs = np.fft.fftshift(brf)
    win = size_equalizer(
        np.outer(np.hamming(winsize), np.hamming(winsize)),
        [rows,cols]
    )
    win2 = size_equalizer(
        np.outer(np.hamming(10), np.ones(cols)),
        [rows,cols]
    )
    brfsw = brfs - brfs * win
    brfsw -= brfsw * win2
    br2 = np.real(np.fft.ifft2(np.fft.ifftshift(brfsw)))
    br2 = br2 * (br2>0)
    return br2

def ncc(br, feature=None):
    if feature is None:
        feature = liner(angle=orientation_finder(br), length=7)
    nccd = np.zeros_like(br)
    pr, pc = feature.shape
    br2 = size_equalizer(
        br.copy(),
        [br.shape[0]+feature.shape[0], br.shape[1]+feature.shape[1]],
        val=np.nan
    )
    br2 = np.ma.array(
        br2,
        mask = np.isnan(br2)
    )
    for i in range(br.shape[0]):
        for j in range(br.shape[1]):
            p1 = br2[i:i+pr, j:j+pc]
            nccd[i,j] = correlation_coefficient(
                p1[p1.mask==False],
                feature[p1.mask==False]
            )
    filter = convolve2d(nccd>0.48, feature, mode='same') > 0
    filter *= (br>5)
    return nccd, feature, filter

def connector(br, n):
    thr = (br>5).astype(np.uint8)
    o = cv2.connectedComponentsWithStats(thr, 8, cv2.CV_32S)
    ind = np.where(o[2][:,-1]>=n)[0]
    a = np.zeros_like(br)
    for i in ind[1:]:
        a += o[1]==i
    return a


# br_orig = np.load('/home/kamo/MAS/stars.npy')
def mf_filt(br, angle=None):
    if angle is None:
        angle = orientation_finder(br)
    print('Angle:{:.2f} - {:.2f}'.format(angle/np.pi*180., angle))
    feature = liner(angle=angle, length=10)
    feature /= np.sum(feature)
    diff_ker = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])
    brd1 = correlate(br, diff_ker, mode='same')
    feature_d = correlate(feature, diff_ker, mode='same')

    match_filtered_d = correlate(brd1, feature_d, mode='same')
    filter = (match_filtered_d>200) * (br > 50)
    filter2 = filter + (br>400)
    br_filtered = br.copy()
    br_filtered[filter2] = 0

    # plt.figure()
    # plt.imshow(feature)
    # plt.title('Feature')
    # plt.show()
    # plt.figure()
    # plt.imshow(br, vmax=40, vmin=0)
    # plt.title('Image')
    # plt.show()
    # plt.figure()
    # plt.imshow(match_filtered_d, vmax=200, vmin=0)
    # plt.title('Match Filtered')
    # plt.show()
    # plt.figure()
    # plt.imshow(filter2)
    # plt.title('Final Filter')
    # plt.show()
    # plt.figure()
    # plt.imshow(br_filtered)
    # plt.title('Final Filtered')
    # plt.show()
    # return match_filtered_d, filter2
    return filter2

def medfilt3d(br, br_err, threshold=50, win_size=(5,10,10)):
    '''
    Code that performs star removal on an orbit of nighttime l1 brightness
    profiles.
    It operates on 3d images (stripe, epoch, altitude) rather than  individual
    1d (altitude) profiles to exploit the information to its best. The star
    detection works based on sliding a 3d window in the data and comparing the
    center element's value with the median of the window. If the value is above
    a threshold, the center pixel is marked as a star and its value is replaced
    with the median of that window. Window size and threshold are determined
    empirically.
    INPUTS:
        br - (6,night_ind,256) array of brightness profile for a particular orbit
        br_err - (6,night_ind,256) array of uncertainty profile for a particular orbit
    OUTPUTS:
        br_corrected - star removed and hot pixel corrected brightness profile
        br_err_modified - modified uncertainty profile according to the correction
    '''
    br = br.copy()
    br_med = median_filter(br, size=win_size)
    br_diff = br - br_med
    filter = br_diff > threshold
    br_filt = br.copy()
    ind = (filter==1)
    br_filt[ind] = br_med[ind]
    br_err[ind] = np.maximum(np.sqrt(br_filt[ind]), br_err[ind])
    return filter, br_filt, br_err

def outlier3d(br=None):
    br = br.copy()
    if np.any(np.isnan(br)):
        br = np.ma.array(br, mask=np.isnan(br))
    br2 = shiftappend(br, w=[1,1])
    br_med = np.median(br2, axis=0)
    # br_med = np.repeat(br_med[np.newaxis,:], 6, axis=0)
    br_diff = br - br_med
    filter = br_diff > 50
    # br_thre = br_med + br_med * 10
    # filter = br > br_thre
    br_filtered = br.copy()
    br_filtered[filter] = 0
    # plt.close('all')
    # for i in range(6):
    #     plt.figure()
    #     plt.imshow(br[i], vmax=400)
    #     plt.title('br[{}]'.format(i))
    #     plt.show()
    #     plt.figure()
    #     plt.imshow(filter[i])
    #     plt.title('filter[{}]'.format(i))
    #     plt.show()
    #     plt.figure()
    #     plt.imshow(br_filtered[i])
    #     plt.title('br_filt[{}]'.format(i))
    #     plt.show()
    return br_filtered, filter

def canny(br, x, y):
    br = br.copy()
    br -= br.min()
    br = br / br.max() * 256
    br = br.astype(np.uint8)
    filter = cv2.Canny(br, x, y)
    br_filtered = br.copy()
    br_filtered[filter>0] = 0
    # plt.figure()
    # plt.imshow(br, vmax=200)
    # plt.title('br')
    # plt.show()
    plt.figure()
    plt.imshow(filter)
    plt.title('filter')
    plt.show()
    # plt.figure()
    # plt.imshow(br_filtered)
    # plt.title('br_filt')
    # plt.show()
    return br_filtered, filter
