# from airglow.FUV_L2 import CreateSummaryPlot
from iconfuv.plotting import tohban2, tohban3, tohban_l1
from iconfuv.misc import lastfile
import numpy as np
import matplotlib.pyplot as plt
import os, sys, glob, netCDF4
path_dir = '/home/kamo/resources/icon-fuv/ncfiles/'

def plotter(layout, date, epoch=None, path_fig=None):
    if layout=='mode_l1':
        file_l1_raw = lastfile(path_dir+'l1/ICON_L1_FUV_SWP_{}_v03r*'.format(date))
        path = path_dir + 'figures/mode_l1/{}.png'.format(date)
        l1 = netCDF4.Dataset(file_l1_raw, mode='r')
        plt.plot(l1.variables['ICON_L1_FUV_Mode'][:])
        plt.plot(l1.variables['ICON_L1_FUVA_SWP_Quality_Flag'][:])
        plt.title('{}\n1:Day 2:Night 3:Cal 4:Nadir 5:Conj 6:Star 7:Ram 8:Off'.format(date))
        plt.ylim([-0.4, 8.4])
        plt.savefig(path)
        plt.close()
        l1.close()
    if layout=='toh':
        file_l2 = lastfile(path_dir+'l2/ICON_L2-5_FUV_Night_{}_v03r101*'.format(date))
        path_fig = path_dir + 'figures/tohban/{}/'.format(date)
        if not os.path.isdir(path_fig):
            os.mkdir(path_fig)
        i=0
        while i<50:
            i += 1
            if not os.path.isdir(path_fig+'v{}/'.format(i)):
                path_fig += 'v{}/'.format(i)
                os.mkdir(path_fig)
                break
        for i in range(6):
            os.mkdir('{}/stripe{}'.format(path_fig,i))
            print('stripe: {}'.format(i+1))
            CreateSummaryPlot(
                # file_netcdf=path_dir+'l2/ICON_L2-5_FUV_Night_{}_v01r000.NC'.format(date),
                file_netcdf=file_l2,
                png_stub=path_fig+'stripe%d/'%i+file_l2[-41:-3]+'_str%d'%i+'.png',
                stripe=i,
                max_ne=None
            )
    elif layout=='tohx':
        file_l1_raw = lastfile(path_dir + 'l1/ICON_L1_FUV_SWP_{}_v05r*'.format(date))
        file_l2 = lastfile(path_dir+'l2/ICON_L2-5_FUV_Night_{}_v05r*'.format(date))
        path_fig = path_dir + 'figures/tohban_x/{}/'.format(date)
        if not os.path.isdir(path_fig):
            os.mkdir(path_fig)
        i=0
        while i<50:
            i += 1
            if not os.path.isdir(path_fig+'v{}/'.format(i)):
                path_fig += 'v{}/'.format(i)
                os.mkdir(path_fig)
                break
        for i in range(6):
            os.mkdir('{}/stripe{}'.format(path_fig,i))
            print('stripe: {}'.format(i+1))
            tohban2(
                # file_l1=path_dir+'l1/ICON_L1_FUV_SWP_{}_v77r000.NC'.format(date),
                # file_l2=path_dir+'l2/ICON_L2-5_FUV_Night_{}_v01r001.NC'.format(date),
                file_l1=file_l1_raw,
                file_l2=file_l2,
                png_stub=path_fig+'stripe%d/'%i+file_l2[-41:-3]+'_str%d'%i+'.png',
                stripe=i,
                # min_br=-5,
                # max_br=50
            )
    elif layout=='tohz':
        file_l1_raw = lastfile(path_dir + 'l1/ICON_L1_FUV_SWP_{}_v05r*'.format(date))
        file_l2 = lastfile(path_dir+'l2/2022/ICON_L2-5_FUV_Night_{}_v05r*'.format(date))
        stripe = 5
        if path_fig is None:
            path_fig = path_dir + 'figures/tohban_z/{}/'.format(date)
            if not os.path.isdir(path_fig):
                os.mkdir(path_fig)
            i=0
            while i<50:
                i += 1
                if not os.path.isdir(path_fig+'v{}/'.format(i)):
                    path_fig += 'v{}/'.format(i)
                    os.mkdir(path_fig)
                    break
            path_fig += '{}/stripe{}'.format(path_fig,stripe)
            os.mkdir(path_fig)

        print('stripe: {}'.format(stripe+1))
        tohban3(
            # file_l1=path_dir+'l1/ICON_L1_FUV_SWP_{}_v77r000.NC'.format(date),
            # file_l2=path_dir+'l2/ICON_L2-5_FUV_Night_{}_v01r001.NC'.format(date),
            file_l1=file_l1_raw,
            flagging=True,
            file_l2=file_l2,
            epoch=epoch,
            png_stub=path_fig+'/'+file_l2[-41:-3]+'_str%d'%stripe+'.png',
            stripe=stripe,
            # min_br=-5,
            # max_br=50
        )
    elif layout=='tohl1':
        file_l1 = lastfile(path_dir + 'l1/ICON_L1_FUV_SWP_{}_v90r*'.format(date))
        path_fig = path_dir + 'figures/tohban_l1/{}/'.format(date)
        if not os.path.isdir(path_fig):
            os.mkdir(path_fig)
        i=0
        while i<50:
            i += 1
            if not os.path.isdir(path_fig+'v{}/'.format(i)):
                path_fig += 'v{}/'.format(i)
                os.mkdir(path_fig)
                break
        tohban_l1(
            file_l1=file_l1,
            png_dir=path_fig,
            stripes=[3],
            both=True
        )


if __name__== "__main__":
    # plotter(str(sys.argv[1]), str(sys.argv[2]))
    path_fig = path_dir+'figures/tohban_z/low_hmf2_cases/'
    plotter('tohz', '2022-06-26', path_fig=path_fig)
    # plotter('tohz', '2022-01-01', 100, path_fig=path_fig)
    # plotter('tohz', '2022-03-31', 1070, path_fig=path_fig)
    # plotter('tohz', '2022-04-10', 470, path_fig=path_fig)
    # plotter('tohz', '2022-05-15', 1000, path_fig=path_fig)
    # plotter('tohz', '2022-05-22', 630, path_fig=path_fig)
    # plotter('tohz', '2022-07-02', 770, path_fig=path_fig)
    # plotter('tohz', '2022-07-04', 430, path_fig=path_fig)
    # plotter('tohz', '2022-08-18', 670, path_fig=path_fig)
    # plotter('tohz', '2022-08-29', 530, path_fig=path_fig)
