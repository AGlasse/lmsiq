#!/usr/bin/env python
"""
Decorators for use in all LMS projects.  Currently just includes @debug

@author: Alistair Glasse

Update:
"""
import astropy.units as u
import numpy as np
from astropy.convolution import Box1DKernel
from scipy.signal import convolve2d
from lmssim_model import Model
from lms_globals import Globals
from lmsaiv_opt_tools import OptTools
from lmsaiv_plot import Plot
from lms_filer import Filer
from lmsdist_util import Util

class Rad04:

    def __init__(self):
        _ = Model()
        return

    @staticmethod
    def pix_flat(title, as_built, **kwargs):
        """ The pixel flat is derived from flood illuminated images by calculating the ratio of the per pixel signal
        to the mean 5x5 box smoothed value centred on it.  We may look at using a best fit 2d surface.
        Valid pixels are identified by setting all pixels below 10 % of the image median to Nan.
        """
        opticon = Globals.nominal
        cut_level = 0.05
        box_hw = 2                          # Smoothing box half width, width = 2 x box_hw + 1
        simulate_pix_flat = True            # True= generate fake dot pattern flat to test

        bs = 2 * box_hw + 1
        boxcar = np.ones((bs, bs))
        opticon_tag = '_' + opticon[0:3]
        inc_tags = ['lms_rad_04', '_flat', opticon_tag]
        mosaics = Filer.read_mosaic_list(Filer.test_data_folder, inc_tags)
        # pix_flat_tag = 'pix_flat' + opticon_tag
        box_area = bs * bs
        cnv_weights = np.full(Globals.det_format, box_area)     # Weights for boxcar smooth
        nr, nc = Globals.det_format
        for b in range(0, box_hw):
            cnv_weights[0:b+1, :] -= bs
            cnv_weights[nr-b-1:nr, :] -= bs
            cnv_weights[:, 0:b+1] -= bs
            cnv_weights[:, nr-b-1:nr] -= bs

        # We generate a test pixel flat here.  Note that the pixel flat is the pixel to pixel relative gain, so
        # we inject a pattern of individual pixels with specified gain.  Large scale structures need to be captured in
        # the 'sky' flat. Here, we create a 'spot' containing a scatterring of 'n_sgp' pixels with specific gain.
        sim_flat = None
        if simulate_pix_flat:
            sim_flat = {}
            spot_radius = 200
            spot_size = 2*spot_radius, 2*spot_radius
            sgp_gain = 1.1

            spot = np.full(spot_size, 1.0)
            for row in range(0, spot_size[0], 30):
                for col in range(0, spot_size[1], 30):
                    spot[row, col] = sgp_gain

            for det_idx in range(0, 4):
                det_no = det_idx + 1
                sim_flat[det_no] = np.full((nr, nc), 1.0)
                for i_det in range(0, det_no):      # One spot per detector
                    row_cen = int(nr / 2)
                    col_cen = int(nc / 2) + int(400 * (i_det - 1.5))
                    r1, r2 = row_cen - spot_radius, row_cen + spot_radius
                    c1, c2 = col_cen - spot_radius, col_cen + spot_radius
                    sim_flat[det_no][r1:r2, c1:c2] = spot

        pix_flat_stack = []
        for mosaic in mosaics[0:4]:
            if Globals.is_debug('high'):
                Plot.mosaic(mosaic)

            pix_flat_name = mosaic.name + '_pix_flat'
            pix_flat = mosaic.copy(clear_data=True, copy_name=pix_flat_name)
            print('Analysing image file= ', mosaic.name)
            for hdu in mosaic.hdu_list:
                det_no = int(hdu.header['ID'])
                det_idx = det_no - 1

                img = np.array(hdu.data)            # Copy the image data
                if sim_flat is not None:
                    img *= sim_flat[det_no]

                img_median = np.nanmedian(img)
                z_cut = cut_level * img_median
                img_thresh = np.where(img < z_cut, np.nan, img)

                img_cnv = convolve2d(img_thresh, boxcar, mode='full')
                nr_cnv, nc_cnv = img_cnv.shape
                img_pf = cnv_weights * img_thresh / img_cnv[box_hw:nr_cnv-box_hw, box_hw:nc_cnv-box_hw]
                pix_flat.hdu_list[det_idx].data = img_pf
                print('Calculated pixel flat for detector ', det_no)
            if Globals.is_debug('high'):
                Plot.mosaic(pix_flat, vmin=0.99, vmax=1.01)
                Plot.histograms(pix_flat, range=(0.99, 1.01), make_labels=True)
            pix_flat_stack.append(pix_flat)
        # Average all pixel flats
        super_pix_flat = None
        n_pix_flats = len(pix_flat_stack)
        for pix_flat in pix_flat_stack:
            if super_pix_flat is None:
                super_pix_flat = pix_flat.copy(copy_name='LMS_RAD_04_pixel_flat' + opticon_tag)
            else:
                super_pix_flat.add(pix_flat)
        super_pix_flat.scale(1./ n_pix_flats)
        Plot.mosaic(super_pix_flat, vmin=0.99, vmax=1.01)
        Plot.histograms(super_pix_flat, range=(0.99, 1.01), make_labels=True)

        return
