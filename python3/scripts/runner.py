import sys
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from airglow.FUV_L2 import Get_lvl2_5_product
import datetime

path_dir = '/home/kamo/resources/iconfuv/nc_files/'

def runner(year, days):
    date = datetime.datetime(year, 1, 1) + datetime.timedelta(days - 1)
    date_str = '{}-{:02d}-{:02d}'.format(date.year, date.month, date.day)
    Get_lvl2_5_product(
            file_input = path_dir + 'ICON_L1_FUV_SWP_{}_v03r000.NC'.format(date_str),
            file_ancillary = path_dir + 'ICON_L0P_FUV_Ancillary_{}_v01r000.NC'.format(date_str),
            file_output = path_dir + 'ICON_L2-5_FUV_Night_{}_v94r000.NC'.format(date_str),
            file_GPI = path_dir + 'ICON_Ancillary_GPI_2015-001-to-2020-044_v01r000.NC',
            Spherical = True,
            regu_order = 2
    )

if __name__== "__main__":
    runner(int(sys.argv[1]), int(sys.argv[2]))
