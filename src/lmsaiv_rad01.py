#!/usr/bin/env python
"""
Decorators for use in all LMS projects.  Currently just includes @debug

@author: Alistair Glasse

Update:
"""
import math
import astropy.units as u
import numpy as np

from lmssim_model import Model
from lms_globals import Globals
from lmsaiv_opt_tools import OptTools
from lmsaiv_plot import Plot
from lms_filer import Filer

class Rad01:

    def __init__(self):
        _ = Model()
        return

    @staticmethod
    def dark(title, as_built, **kwargs):
        """ Field of view calculation using flood illuminated continuum spectral images.  Populates the slice bounds
        map in the AsBuilt object
        """
        Rad01.analyse_darks()
        print('Done')
        return as_built

    @staticmethod
    def analyse_darks():
        inc_tags = ['lms_rad_01', '_dark']
        darks = Filer.read_mosaic_list(Filer.test_data_folder, inc_tags)
        Rad01.dark_stats(darks)
        if Globals.is_debug('low'):
            title = 'Dark'
            Plot.mosaic(darks[0], title=title)
            Plot.histograms(darks[0])
        return

    @staticmethod
    def dark_stats(mosaics):
        for mosaic in mosaics:
            dit = mosaic.primary_hdr['HIERARCH ESO DET DIT']
            ndit = mosaic.primary_hdr['HIERARCH ESO DET NDIT']
            t_int = dit * ndit

            print()
            print("File = {:s}".format(mosaic.name))
            fmt = "Signal distribution statistics, ndit= {:d} x dit= {:3.1f} sec, integration time = {:6.1f}"
            print(fmt.format(ndit, dit, t_int))
            fmt = "{:>8s},{:>10s},{:>10s},{:>10s},{:>10s}"
            print(fmt.format('Detector', 'median', 'stdev', 'median', 'Rd_Noise'))
            print(fmt.format('No.', 'DN', 'DN', 'el/sec.', 'el.'))
            fmt = "{:8d},{:10.3f},{:10.3f},{:10.3f},{:10.3f}"

            for i, hdu in enumerate(mosaic.hdu_list):
                el_adu = float(hdu.header['HIERARCH ESO DET3 CHIP GAIN'])
                median = np.median(hdu.data)
                stdev = np.std(hdu.data)
                median_current = median * el_adu / dit
                rd_noise = stdev * el_adu / math.sqrt(2. / ndit)
                text = fmt.format(i + 1, median, stdev, median_current, rd_noise)
                print(text)
        return

    @staticmethod
    def _find_rrf(flood, slice_map):
        # Generate relative response tuple.
        cols = np.arange(0, 4096, 1)
        rrf = slice_map.copy(copy_name='rel_res_function')

        wave_mosaic_cen = flood.primary_hdr['HIERARCH ESO INS WLEN CEN'] * u.micron
        for i in range(0, 4):
            slice_map_data = slice_map.hdu_list[i].data
            slice_mask = np.where(slice_map_data > 0., 1., 0.)
            # Very approximate dispersion...!
            hdr = flood.hdu_list[i].header
            flood_image = flood.hdu_list[i].data
            x_det_cen = float(hdr['X_CEN']) * u.mm
            n_det_rows, n_det_cols = flood_image.shape
            pix_size = float(hdr['HIERARCH AIT PIXEL_PITCH']) * u.mm
            c_det_cen = x_det_cen / pix_size
            c_det_org = c_det_cen - n_det_cols / 2
            disp = .08 * u.micron / (2. * n_det_cols)
            waves = wave_mosaic_cen + disp * (c_det_org + cols)
            flux = Model.black_body(waves, tbb=1000.)
            rrf_image = rrf.hdu_list[i].data
            for row in range(0, n_det_rows):
                idx = np.argwhere(slice_mask[row] > 0.)
                rrf_image[row, idx] = flood_image[row, idx] / flux[idx]
            rrf.hdu_list[i].data = rrf_image
        Plot.mosaic(rrf, title='Rel Response Function', cmap='grey', mask=(0.0, 'black'))
        return

    @staticmethod
    def _parse_abo(file_name):
        """ Extract alpha, beta and observation number from a fits file name.  Used in lms_opt_01_t2
        """
        signed = {'_a': True, '_b': True, '_o': False}
        ip = file_name.find('_grid') + 5        # Get start position of wavelength, alpha and beta sub strings
        abo = []
        for tag in signed:
            ip = file_name.find(tag, ip) + len(tag)
            sign = 1.0
            if signed[tag]:
                sign = 1. if file_name[ip] == 'p' else -1.
                ip += 1
            mag = float(file_name[ip: ip + 3])
            val = sign * mag
            abo.append(val)
        return tuple(abo)

