import math
import copy
import numpy as np
from astropy import units as u
from lms_globals import Globals


class OptTools:


    def __init__(self):
        return

    @staticmethod
    def dark_stats(mosaics):
        for mosaic in mosaics:
            file_name, hdr, hdus = mosaic
            dit = hdr['HIERARCH ESO DET DIT']
            ndit = hdr['HIERARCH ESO DET NDIT']
            t_int = dit * ndit

            print()
            print("File = {:s}".format(file_name))
            print("Signal distribution statistics, integration time = {:10.1f}".format(t_int))
            fmt = "{:>8s},{:>10s},{:>10s},{:>10s},{:>10s}"
            print(fmt.format('Detector', 'median', 'stdev', 'median', 'Rd_Noise'))
            print(fmt.format('No.', 'ADU', 'ADU', 'el/sec.', 'el.'))
            fmt = "{:8d},{:10.3f},{:10.3f},{:10.3f},{:10.3f}"

            for i, hdu in enumerate(hdus):
                el_adu = hdu.header['HIERARCH ESO DET3 CHIP GAIN']
                median = np.median(hdu.data)
                stdev = np.std(hdu.data)
                median_current = median * el_adu / dit
                rd_noise = stdev * el_adu / math.sqrt(2. / ndit)
                text = fmt.format(i + 1, median, stdev, median_current, rd_noise)
                print(text)
        return

    @staticmethod
    def extract_alpha_traces(mosaic, col_spacing=100):
        """ For a spectral image of a compact source, extract a set of along-column profiles
        col_spacing: column gap between profiles.

        """
        mos_name, mos_primary_header, mos_hdus = mosaic
        trace = None
        alpha_det = [0.]*4
        for i, hdu in enumerate(mos_hdus):
            col = 400
            image = hdu.data
            y_vals = image[:, col]
            x_vals = np.arange(0, y_vals.shape[0])
            pts_list = []
            profile = 'bum', x_vals, y_vals, pts_list

            title, x_val, y_val, pts_list = profile
            nob = 1
            # faint_level = np.nanpercentile(profile, dark_pctile)
            # bright_level = np.nanpercentile(profile, bright_pctile)
            # on_rows, off_rows, slice_bounds = [], [], []
            # row_off = 0
        return [profile]

