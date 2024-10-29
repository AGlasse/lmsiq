#!/usr/bin/env python
"""

"""
import math

import numpy as np
from astropy.io import fits

from lmsdist_util import Util
from lms_wcal import Wcal
from lms_globals import Globals
from lms_efficiency import Efficiency
from lmsdist_plot import Plot
from lms_filer import Filer
from lmsiq_image_manager import ImageManager

class ToySim:

    tau_blaze_kernel = None     # Kernel blaze profile tau(x) where x = (wave / blaze_wave(eo) - 1)

    def __init__(self):
        return

    def load_psf_dict(self, opticon, ech_ord, downsample=False, slice_no_tgt=13):
        analysis_type = 'iq'

        nominal = Globals.nominal
        nom_iq_date_stamp = '2024073000'
        nom_config = (analysis_type, nominal, nom_iq_date_stamp,
                      'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
                      None, None)

        spifu = Globals.spifu
        spifu_date_stamp = '20231009'
        spifu_config = (analysis_type, spifu, spifu_date_stamp,
                        'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                        None, None)

        model_configurations = {nominal: nom_config, spifu: spifu_config}
        model_config = model_configurations[opticon]
        filer = Filer(model_config)

        _, _, date_stamp, _, _, _ = model_config
        dataset_folder = '../data/iq/nominal/' + date_stamp + '/'
        config_no = 41 - ech_ord
        config_str = "_config{:03d}".format(config_no)
        field_str = "_field{:03d}".format(1)
        defoc_str = '_defoc000um'
        config_str = 'lms_' + date_stamp + config_str + field_str + defoc_str
        iq_folder = dataset_folder + config_str + '/'
        psf_dict = {}
        psf_sum = 0.

        sn_min, sn_max = slice_no_tgt - 4, slice_no_tgt + 5
        for slice_no in range(sn_min, sn_max):
            slice_no_offset = slice_no_tgt - slice_no
            iq_slice_str = "_spat{:02d}".format(slice_no) + '_spec0_detdesi'
            iq_filename = config_str + iq_slice_str + '.fits'
            iq_path = iq_folder + iq_filename
            hdr, psf = filer.read_fits(iq_path)
            # print("slice_no={:d}, psf_max={:10.3e}".format(slice_no, np.amax(psf)))
            if downsample:
                oversampling = 4
                n_psf_rows, n_psf_ncols = psf.shape
                n_ds_rows, n_ds_cols = int(n_psf_rows / oversampling), int(n_psf_ncols / oversampling)
                psf = psf.reshape(n_ds_rows, oversampling, n_ds_cols, -1).mean(axis=3).mean(axis=1)   # down sample
            psf_dict[slice_no_offset] = hdr, psf
            psf_sum += np.sum(psf)
        # Normalise the PSFs to have unity total flux in detector space
        for slice_no in range(sn_min, sn_max):
            slice_no_offset = slice_no_tgt - slice_no
            _, psf = psf_dict[slice_no_offset]
            norm_factor = oversampling * oversampling / psf_sum
            psf *= norm_factor
        return psf_dict

    @staticmethod
    def add_psf(psf, det_point, det_imgs):
        oversampling = Globals.oversampling
        det_idx, det_x, det_y = det_point
        _, n_psfcols = psf.shape
        nr_psf, nc_psf = psf.shape
        hw_det_psf = nc_psf // (2 * oversampling)  # Half-width of PSF image on detector
        is_oob = det_x < 0. or det_x > 2048.
        if is_oob:
            return is_oob, det_imgs
        det_img = det_imgs[det_idx]
        _, n_imcols = det_img.shape

        # Add the PSF to the image at this location at the PSF resolution (x4 image pix/det pix)
        r1, c1 = int(det_y - hw_det_psf), int(det_x - hw_det_psf)
        c1 = 0 if c1 < 0 else c1
        fw_det_psf = 2 * hw_det_psf
        r2, c2 = r1 + fw_det_psf, c1 + fw_det_psf
        pc1, pc2 = 0, n_psfcols
        is_oob = c1 >= n_imcols or c2 < 1
        if is_oob:
            return is_oob, det_imgs
        if c1 < 0:
            pc1 = -c1 * oversampling
            c1 = 0
        if c2 > n_imcols:
            ncr = c2 - n_imcols
            pc2 = n_psfcols - ncr * oversampling
            c2 = n_imcols
        # down-sample psf and write back into image
        psf_sub = psf[:, pc1:pc2]
        nss_rows = psf_sub.shape[0] // oversampling
        nss_cols = psf_sub.shape[1] // oversampling
        psf_ds = psf_sub.reshape((nss_rows, oversampling, nss_cols, -1)).mean(axis=3).mean(axis=1)
        det_imgs[det_idx][r1:r2, c1:c2] += psf_ds
        return is_oob, det_imgs

    @staticmethod
    def drizzle_psf(psf, det_point, det_imgs, det_hits):
        oversampling = Globals.oversampling
        det_idx, det_x, det_y = det_point
        _, n_psfcols = psf.shape
        nr_psf, nc_psf = psf.shape
        hw_det_psf = nc_psf // (2 * oversampling)  # Half-width of PSF image on detector
        is_oob = det_x < 0. or det_x > 2048.
        if is_oob:
            return is_oob, det_imgs
        det_img = det_imgs[det_idx]
        _, n_imcols = det_img.shape

        # Add the PSF to the image at this location at the PSF resolution (x4 image pix/det pix)
        r1, c1 = int(det_y - hw_det_psf), int(det_x - hw_det_psf)
        c1 = 0 if c1 < 0 else c1
        fw_det_psf = 2 * hw_det_psf
        r2, c2 = r1 + fw_det_psf, c1 + fw_det_psf
        pc1, pc2 = 0, n_psfcols
        is_oob = c1 >= n_imcols or c2 < 1
        if is_oob:
            return is_oob, det_imgs
        if c1 < 0:
            pc1 = -c1 * oversampling
            c1 = 0
        if c2 > n_imcols:
            ncr = c2 - n_imcols
            pc2 = n_psfcols - ncr * oversampling
            c2 = n_imcols
        # down-sample psf and write back into image
        psf_sub = psf[:, pc1:pc2]
        nss_rows = psf_sub.shape[0] // oversampling
        nss_cols = psf_sub.shape[1] // oversampling
        psf_ds = psf_sub.reshape((nss_rows, oversampling, nss_cols, -1)).mean(axis=3).mean(axis=1)
        det_imgs[det_idx][r1:r2, c1:c2] += psf_ds
        return is_oob, det_imgs

    @staticmethod
    def build_bb_emission(wave_range=None, tbb=1000.):
        hp = 6.626e-34
        cc = 2.997e+8
        kb = 1.38e-23
        as2_sterad = 4.25e10
        m_um = 1.e-6
        wmin, wmax = wave_range[0], wave_range[1]
        delta_w = wmin / 100000
        waves = np.arange(wmin, wmax, delta_w)
        flux = np.zeros(waves.shape)
        wm = waves * m_um
        a = hp * cc / (wm * kb * tbb)
        inz = np.argwhere(a < 400.)         # Set UV catastrophe points to zero
        b = np.exp(a[inz]) - 1.
        c = 2 * cc * np.power(wm[inz], -4)
        flux[inz] = m_um * (c / b) / as2_sterad

        units = 'phot/s/m2/um/arcsec2'
        return waves, flux, units

    @staticmethod
    def make_tau_blaze(n_ech):
        """ Generate a blaze profile (wavelength v efficiency) for an echelle order.
        """
        xwid = 0.042
        n_pts = 100
        if ToySim.tau_blaze_kernel is None:
            x = np.linspace(-5.*xwid, +5.*xwid, n_pts)
            tk = 0.7 * np.power(np.sinc(x / xwid), 2)
            ToySim.tau_blaze_kernel = x, tk
        x, tk = ToySim.tau_blaze_kernel
        wave_1 = Globals.wav_first_order          # First order blaze wavelength
        wave_n = wave_1 / n_ech
        w_wid = wave_n * xwid
        w_lo = wave_n - 5. * w_wid
        w_hi = wave_n + 5. * w_wid
        waves = np.linspace(w_lo, w_hi, n_pts)
        return waves, tk

    @staticmethod
    def load_sky_emission(waves_in=None, wave_range=None):
        path = '../data/elt_sky.fits'
        hdu_list = fits.open(path, mode='readonly')
        data_table = hdu_list[1].data
        waves_all = data_table['lam'] / 1000.
        flux_all, flux_errs = data_table['flux'], None
        flux_units, label, colour = 'ph/s/m2/um/arcsec2', 'Emission', 'orange'
        print("Loaded sky transmission and emission spectrum with units {:s}".format(flux_units))
        if waves_in is not None:        # Interpolate spectrum onto input wavelength list
            flux_new = np.zeros(waves_in.shape)
            i = 0
            fmt = "{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}"
            print(fmt.format('New Wave', 'New T', 'W1', 'W2', 'T1', 'T2'))
            for j, new_wave in enumerate(waves_in):
                while waves_all[i] < new_wave:
                    i += 1
                flux_new[j] = np.interp(new_wave, waves_all[i - 1:i + 1], flux_all[i - 1:i + 1])
                i += 1
            return waves_in, flux_new, flux_units

        if wave_range is not None:
            wmin, wmax = wave_range[0], wave_range[1]
            idx1 = np.argwhere(wmin < waves_all)
            idx2 = np.argwhere(waves_all < wmax)
            idx = np.intersect1d(idx1, idx2)
            waves, flux = waves_all[idx], flux_all[idx]

            return waves, flux, flux_units

        flux_units = 'phot/s/m2/um/arcsec2'
        return waves_all, flux_all, flux_units
