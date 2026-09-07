#!/usr/bin/env python
"""
Decorators for use in all LMS projects.  Currently just includes @debug

@author: Alistair Glasse

Update:
"""
import math

import numpy as np
from lmsaiv_opt_tools import OptTools
from lmsaiv_plot import Plot
from lms_filer import Filer


class Opt03:

    def __init__(self):
        return

    @staticmethod
    def psf(test_name, title, as_built, **kwargs):
        """ Analysis of laser + single pinhole data to characterise the mochromatic point source.
        """
        inc_tags = [test_name, 'nom_dark']
        darks = Filer.read_mosaic_list(inc_tags)
        do_plot = kwargs.get('do_plot', True)
        if do_plot:
            Plot.mosaic(darks[0], title=title)
            Plot.histograms(darks[0])
        OptTools.dark_stats(darks)

        floods = Filer.read_mosaic_list(inc_tags=[test_name, 'flood'])
        for flood in floods:
            slice_map, profiles = OptTools.flood_stats(flood)
            do_plot = True
            if do_plot:
                Plot.profiles(profiles)
                Plot.mosaic(flood, title=title, cmap='hot')        # Use cmap='hot', 'gray' etc.
                Plot.mosaic(slice_map, title='Slice Map', cmap='hsv', mask=(0.0, 'black'))
            as_built['slice_map'] = slice_map

        # Generate relative response tuple.
        cols = np.arange(0, 4096, 1)
        for flood in floods:
            slice_map = as_built['slice_map']
            rrf = OptTools.copy_mosaic(slice_map, copy_name='rel_res_function')
            rrf_name, rrf_primary_header, rrf_hdus = rrf
            Plot.mosaic(slice_map, title='Slice Map', cmap='hsv', mask=(0.0, 'black'))
            name, primary_hdr, hdus = flood
            wave_mosaic_cen = primary_hdr['HIERARCH ESO INS WLEN CEN'] * u.micron
            _, _, slice_map_hdus = slice_map
            for i in range(0, 4):
                slice_map_data = slice_map_hdus[i].data
                slice_mask = np.where(slice_map_data > 0., 1., 0.)
                # Very approximate dispersion...!
                hdr = hdus[i].header
                flood_image = hdus[i].data
                x_det_cen = float(hdr['X_CEN']) * u.mm
                n_det_cols = float(hdr['X_SIZE'])
                pix_size = hdr['HIERARCH pixel_size'] * u.mm
                c_det_cen = x_det_cen / pix_size
                c_det_org = c_det_cen - n_det_cols / 2
                disp = .08 * u.micron / (2. * n_det_cols)
                waves = wave_mosaic_cen + disp * (c_det_org + cols)
                flux = Model.black_body(waves, tbb=1000.)
                n_det_rows = int(hdr['Y_SIZE'])
                rrf_image = rrf_hdus[i].data
                for row in range(0, n_det_rows):
                    idx = np.argwhere(slice_mask[row] > 0.)
                    rrf_image[row, idx] = flood_image[row, idx] / flux[idx]
                rrf_hdus[i].data = rrf_image
            Plot.mosaic(rrf, title='Rel Response Function', cmap='grey', mask=(0.0, 'black'))
        print('Done')
        return as_built

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
