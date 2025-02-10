#!/usr/bin/env python
"""

"""
import math
import numpy as np
from astropy.io import fits
from lmsdist_util import Util
from lms_globals import Globals
from lms_filer import Filer
from lms_detector import Detector


class Model:

    tau_blaze_kernel = None  # Kernel blaze profile tau(x) where x = (wave / blaze_wave(eo) - 1)

    tau_cfo_mask = .09         # CFO pinhole mask to detector (excluding detector and echelle blaze profile)
    tau_wcu_fp = .05
    tau_wcu_hs = .005                    # WCU Integrating sphere to detector
    tau_sky = .02

    # Define a list of extended illumination sources.  These images will be convolved with the 'target slice' PSF.
    bgd_srcs = {'wcu_bb': {'sed': 'bb', 'temperature': 1000., 'tau': tau_wcu_hs},               # 1000 K black body
                'cfo_mask': {'sed': 'bb', 'temperature': 70., 'tau': tau_cfo_mask},
                'wcu_mask': {'sed': 'bb', 'temperature': 300., 'tau': tau_wcu_fp},
                'wcu_ls': {'sed': 'laser', 'flux': 1.E+09, 'wavelength': 3390, 'tau': tau_wcu_hs},
                'wcu_ll': {'sed': 'laser', 'flux': 1.E+09, 'wavelength': 5240, 'tau': tau_wcu_hs},
                'wcu_lt': {'sed': 'laser', 'flux': 1.E+09, 'nlines': 27, 'wshort': 4590, 'wlong': 4850, 'tau': tau_wcu_hs},
                'sky': {'sed': 'sky', 'tau': tau_sky}    # Model sky emission spectrum
                }

    # Define one or more (point-like) pinhole masks which will spatially filter the extended source.  The model
    # specified PSFs at +-4 slices from the target slice will be convolved with the 'pinhole' images.
    fp_masks = {'cfo': {'id': 'cfo', 'efp_x': 0., 'efp_y': 0.,           # On-axis pinhole in boresight
                        'mask_ext': 'cfo_mask'},
                'wcu': {'id': 'wcu', 'efp_x': 0., 'efp_y': 0.,           # Steerable pinhole in WCU.
                        'mask_ext': 'wcu_amb'},
                'open': {'id': 'open', 'efp_x': None, 'efp_y': None,     # FP-1 open position
                         'mask_ext': 'none'}
                }

    def __init__(self):
        return

    def __str__(self):
        txt = 'Background sources, '
        for key in self.bgd_srcs:
            txt += "{:s}, ".format(key)
        txt += '\n'
        txt += 'Focal plane masks, '
        for key in self.fp_masks:
            txt += "{:s}, ".format(key)
        return txt

    @staticmethod
    def _make_waves(wbounds):
        wmin, wmax = wbounds[0], wbounds[1]
        delta_w = wmin / 200000
        waves = np.arange(wmin, wmax, delta_w)
        return waves

    def get_pnh_flux(self, wbounds, mask):
        fp_mask = self.fp_masks[mask]
        src = fp_mask['mask_ext']
        return self.get_flux(wbounds, src)

    def get_flux(self, wbounds, src, n_lines=1):
        """ Load selected extended background spectrum (units ph/s/m2/as2/um) for wavelength range which overfills mosaic
        f_units_ext_in = 'phot/s/m2/um/arcsec2'
        """
        w_ext = self._make_waves(wbounds)
        f_ext_in, f_units_ext_in = None, None
        source = self.bgd_srcs[src]
        sed = source['sed']
        tau_qe = source['tau'] * Detector.qe
        if sed == 'bb':
            tau_bb = 0.05 * 1.0         # From RvBs flux model
            f_bb, f_units_ext_in = Model.build_bb_emission(w_ext, tbb=source['temperature'])
            f_ext_in = tau_qe * tau_bb * f_bb
        if sed == 'sky':
            f_sky, f_units_ext_in = Model.load_sky_emission(w_ext)
            f_ext_in = tau_qe * f_sky
        if sed == 'laser':
            f_laser, f_units_ext_in = Model.build_laser_emission(w_ext, source)
            f_ext_in = tau_qe * f_laser
        atel = math.pi * (39. / 2)**2               # ELT collecting area
        alpha_pix = Globals.alpha_mas_pix / 1000.   # Along slice pixel scale
        beta_slice = Globals.beta_mas_slice / 1000.   # Slice width
        delta_w = wbounds[0] / 100000               # Spectral resolution
        pix_delta_w = 2.5                           # Pixels per spectral resolution element
        f_ext = f_ext_in * atel * alpha_pix * beta_slice * delta_w / pix_delta_w
        print('- converted to el/sec/pix')
        return w_ext, f_ext, f_units_ext_in

    def make_blaze_dictionary(transforms, opticon):
        blaze = {}
        for key in transforms:
            transform = transforms[key]
            cfg = transform['configuration']
            if cfg['slice'] != 13:
                continue
            # if opticon == Globals.spifu:
            #     if cfg['spifu'] != 1:
            #         continue
            ech_ang = cfg['ech_ang']
            mfp_bs = {'mfp_x': [0.], 'mfp_y': [0.]}
            ech_ord = cfg['ech_ord']
            efp_bs = Util.mfp_to_efp(transform, mfp_bs)
            wave = efp_bs['efp_w'][0]
            if ech_ord not in blaze:
                blaze[ech_ord] = {}
            blaze[ech_ord][ech_ang] = wave
        return blaze

    def get_fp_mask(self, fp_key):
        fp_mask = Model.fp_masks[fp_key]
        return fp_mask

    def load_psf_dict(opticon, ech_ord, downsample=False, slice_no_tgt=13):
        analysis_type = 'iq'

        nominal = Globals.nominal
        nom_iq_date_stamp = '2024073000'
        nom_config = (analysis_type, nominal, nom_iq_date_stamp,
                      'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
                      None, None)

        spifu = Globals.spifu
        spifu_date_stamp = '2024061802'
        spifu_config = (analysis_type, spifu, spifu_date_stamp,
                        'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                        None, None)

        model_configurations = {nominal: nom_config, spifu: spifu_config}
        model_config = model_configurations[opticon]
        filer = Filer(model_config)
        defoc_str = '_defoc000um'

        _, _, date_stamp, _, _, _ = model_config
        dataset_folder = '../data/iq/' + opticon + '/' + date_stamp + '/'
        config_no = 41 - ech_ord if opticon == nominal else 0
        config_str = "_config{:03d}".format(config_no)

        psf_sum = 0.
        # Find a slice number
        nom_slice_no_rep_field = {1: (9, 17)}

        psf_dict = {}  # Create a new set of psfs

        # Use the boresight field position (field_no = 1) for now...
        (fn_min, fn_max) = (1, 2) if opticon == nominal else (1, 4)
        for field_no in range(fn_min, fn_max):
            field_idx = field_no - 1
            field_str = "_field{:03d}".format(field_no)
            iq_folder = 'lms_' + date_stamp + config_str + field_str + defoc_str
            spec_no = 0
            sn_radius = 4 if opticon == nominal else 1
            sn_min, sn_max = slice_no_tgt - sn_radius, slice_no_tgt + sn_radius + 1

            if opticon == spifu:
                # field_idx = field_no - 1
                spec_no = 1
                sn_min = slice_no_tgt - 1 + field_idx % 3
                sn_max = sn_min + 1

            for slice_no in range(sn_min, sn_max):
                iq_slice_str = "_spat{:02d}".format(slice_no) + "_spec{:d}_detdesi".format(spec_no)
                iq_filename = iq_folder + iq_slice_str + '.fits'
                iq_path = iq_folder + '/' + iq_filename
                file_path = dataset_folder + iq_path
                hdr, psf = filer.read_fits(file_path)
                # print("slice_no={:d}, psf_max={:10.3e}".format(slice_no, np.amax(psf)))
                if downsample:
                    oversampling = 4
                    n_psf_rows, n_psf_ncols = psf.shape
                    n_ds_rows, n_ds_cols = int(n_psf_rows / oversampling), int(n_psf_ncols / oversampling)
                    psf = psf.reshape(n_ds_rows, oversampling, n_ds_cols, -1).mean(axis=3).mean(axis=1)   # down sample
                slice_no_offset = slice_no - slice_no_tgt
                psf_dict[slice_no_offset] = hdr, psf
                psf_sum += np.sum(psf)
            # Normalise the PSFs to have unity total flux in detector space
            for slice_no in range(sn_min, sn_max):
                slice_no_offset = slice_no - slice_no_tgt
                _, psf = psf_dict[slice_no_offset]
                norm_factor = oversampling * oversampling / psf_sum
                psf *= norm_factor
        return psf_dict

    def make_blaze_dictionary(transforms, opticon):
        blaze = {}
        for key in transforms:
            transform = transforms[key]
            cfg = transform['configuration']
            if cfg['slice'] != 13:
                continue
            ech_ang = cfg['ech_ang']
            mfp_bs = {'mfp_x': [0.], 'mfp_y': [0.]}
            ech_ord = cfg['ech_ord']
            efp_bs = Util.mfp_to_efp(transform, mfp_bs)
            wave = efp_bs['efp_w'][0]
            if ech_ord not in blaze:
                blaze[ech_ord] = {}
            blaze[ech_ord][ech_ang] = wave
        return blaze

    def load_psf_dict_old(opticon, ech_ord, downsample=False, slice_no_tgt=13):
        analysis_type = 'iq'

        nominal = Globals.nominal
        nom_iq_date_stamp = '2024073000'
        nom_config = (analysis_type, nominal, nom_iq_date_stamp,
                      'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
                      None, None)

        spifu = Globals.spifu
        spifu_date_stamp = '2024061802'
        spifu_config = (analysis_type, spifu, spifu_date_stamp,
                        'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                        None, None)

        model_configurations = {nominal: nom_config, spifu: spifu_config}
        model_config = model_configurations[opticon]
        filer = Filer(model_config)
        defoc_str = '_defoc000um'

        _, _, date_stamp, _, _, _ = model_config
        dataset_folder = '../data/model/iq/' + opticon + '/' + date_stamp + '/'
        config_no = 41 - ech_ord if opticon == nominal else 0
        config_str = "_config{:03d}".format(config_no)

        psf_sum = 0.
        psf_dict = {}  # Create a new set of psfs

        # Use the boresight field position (field_no = 1) for now...
        (fn_min, fn_max) = (1, 2) if opticon == nominal else (1, 4)
        for field_no in range(fn_min, fn_max):
            field_idx = field_no - 1
            field_str = "_field{:03d}".format(field_no)
            iq_folder = 'lms_' + date_stamp + config_str + field_str + defoc_str
            spec_no = 0
            sn_radius = 4 if opticon == nominal else 1
            sn_min, sn_max = slice_no_tgt - sn_radius, slice_no_tgt + sn_radius + 1

            if opticon == spifu:
                # field_idx = field_no - 1
                spec_no = 1
                sn_min = slice_no_tgt - 1 + field_idx % 3
                sn_max = sn_min + 1

            for slice_no in range(sn_min, sn_max):
                iq_slice_str = "_spat{:02d}".format(slice_no) + "_spec{:d}_detdesi".format(spec_no)
                iq_filename = iq_folder + iq_slice_str + '.fits'
                iq_path = iq_folder + '/' + iq_filename
                file_path = dataset_folder + iq_path
                hdr, psf = filer.read_fits(file_path)
                # print("slice_no={:d}, psf_max={:10.3e}".format(slice_no, np.amax(psf)))
                if downsample:
                    oversampling = 4
                    n_psf_rows, n_psf_ncols = psf.shape
                    n_ds_rows, n_ds_cols = int(n_psf_rows / oversampling), int(n_psf_ncols / oversampling)
                    psf = psf.reshape(n_ds_rows, oversampling, n_ds_cols, -1).mean(axis=3).mean(axis=1)  # down sample
                slice_no_offset = slice_no - slice_no_tgt
                psf_dict[slice_no_offset] = hdr, psf
                psf_sum += np.sum(psf)
            # Normalise the PSFs to have unity total flux in detector space
            for slice_no in range(sn_min, sn_max):
                slice_no_offset = slice_no - slice_no_tgt
                _, psf = psf_dict[slice_no_offset]
                norm_factor = oversampling * oversampling / psf_sum
                psf *= norm_factor
        return psf_dict

    @staticmethod
    def build_laser_emission(waves, laser):
        delta_w = .0001          # Laser line width (microns)

        flux = np.zeros(waves.shape)
        # if laser['nlines'] > 1:
        #     line_waves = list(np.linspace(laser['wshort'], laser['wlong'], laser['nlines']))
        # else:
        line_waves = [laser['wavelength']]
        f_laser = laser['flux']
        for line_wave in line_waves:
            woff = np.abs(waves - line_wave / 1000.)
            las_idx = np.argwhere(woff < delta_w)
            flux[las_idx] = f_laser
        units = 'phot/s/m2/um/arcsec2'
        return flux, units

    @staticmethod
    def build_bb_emission(waves, tbb=1000.):
        hp = 6.626e-34
        cc = 2.997e+8
        kb = 1.38e-23
        as2_sterad = 4.25e10
        m_um = 1.e-6
        flux = np.zeros(waves.shape)
        wm = waves * m_um
        a = hp * cc / (wm * kb * tbb)
        inz = np.argwhere(a < 400.)         # Set UV catastrophe points to zero
        b = np.exp(a[inz]) - 1.
        c = 2 * cc * np.power(wm[inz], -4)
        flux[inz] = m_um * (c / b) / as2_sterad
        units = 'phot/s/m2/um/arcsec2'
        return flux, units

    @staticmethod
    def make_tau_blaze(blaze, ech_ord, ech_ang):
        """ Generate a blaze profile (wavelength v efficiency) for an echelle order.
        I = sinc^2(pi (w - w_blaze) / w_width), where w_width = 0.042 w_blaze from SPIE model
        """
        w_n = blaze[ech_ord][ech_ang]
        w_1 = w_n * ech_ord
        n_pts = 500
        w_wid = 0.5 * w_n / (ech_ord + 1)
        w_lo = w_n - 5. * w_wid
        w_hi = w_n + 5. * w_wid
        waves = np.linspace(w_lo, w_hi, n_pts)
        tk = 0.7 * np.power(np.sinc(math.pi * ech_ord * (ech_ord * waves - w_1) / w_1), 2)
        return waves, tk

    def load_psf_dict(opticon, ech_ord, downsample=False, slice_no_tgt=13):
        analysis_type = 'iq'

        nominal = Globals.nominal
        nom_iq_date_stamp = '2024073000'
        nom_config = (analysis_type, nominal, nom_iq_date_stamp,
                      'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
                      None, None)

        spifu = Globals.spifu
        spifu_date_stamp = '2024061802'
        spifu_config = (analysis_type, spifu, spifu_date_stamp,
                        'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                        None, None)

        model_configurations = {nominal: nom_config, spifu: spifu_config}
        model_config = model_configurations[opticon]
        filer = Filer(model_config)
        defoc_str = '_defoc000um'

        _, _, date_stamp, _, _, _ = model_config
        dataset_folder = '../data/model/iq/' + opticon + '/' + date_stamp + '/'
        config_no = 41 - ech_ord if opticon == nominal else 0
        config_str = "_config{:03d}".format(config_no)

        psf_sum = 0.
        # Find a slice number
        nom_slice_no_rep_field = {1: (9, 17)}

        psf_dict = {}  # Create a new set of psfs

        # Use the boresight field position (field_no = 1) for now...
        (fn_min, fn_max) = (1, 2) if opticon == nominal else (1, 4)
        for field_no in range(fn_min, fn_max):
            field_idx = field_no - 1
            field_str = "_field{:03d}".format(field_no)
            iq_folder = 'lms_' + date_stamp + config_str + field_str + defoc_str
            spec_no = 0
            sn_radius = 4 if opticon == nominal else 1
            sn_min, sn_max = slice_no_tgt - sn_radius, slice_no_tgt + sn_radius + 1

            if opticon == spifu:
                # field_idx = field_no - 1
                spec_no = 1
                sn_min = slice_no_tgt - 1 + field_idx % 3
                sn_max = sn_min + 1

            for slice_no in range(sn_min, sn_max):
                iq_slice_str = "_spat{:02d}".format(slice_no) + "_spec{:d}_detdesi".format(spec_no)
                iq_filename = iq_folder + iq_slice_str + '.fits'
                iq_path = iq_folder + '/' + iq_filename
                file_path = dataset_folder + iq_path
                hdr, psf = filer.read_fits(file_path)
                # print("slice_no={:d}, psf_max={:10.3e}".format(slice_no, np.amax(psf)))
                if downsample:
                    oversampling = 4
                    n_psf_rows, n_psf_ncols = psf.shape
                    n_ds_rows, n_ds_cols = int(n_psf_rows / oversampling), int(n_psf_ncols / oversampling)
                    psf = psf.reshape(n_ds_rows, oversampling, n_ds_cols, -1).mean(axis=3).mean(axis=1)  # down sample
                slice_no_offset = slice_no - slice_no_tgt
                psf_dict[slice_no_offset] = hdr, psf
                psf_sum += np.sum(psf)
            # Normalise the PSFs to have unity total flux in detector space
            for slice_no in range(sn_min, sn_max):
                slice_no_offset = slice_no - slice_no_tgt
                _, psf = psf_dict[slice_no_offset]
                norm_factor = oversampling * oversampling / psf_sum
                psf *= norm_factor
        return psf_dict

    @staticmethod
    def load_sky_emission(waves):
        path = '../data/sky/elt_sky.fits'
        hdu_list = fits.open(path, mode='readonly')
        data_table = hdu_list[1].data
        waves_all = data_table['lam'] / 1000.
        flux_all, flux_errs = data_table['flux'], None
        flux_units, label, colour = 'ph/s/m2/um/arcsec2', 'Emission', 'orange'
        print("Loaded sky transmission and emission spectrum with units {:s}".format(flux_units))
        flux_new = np.zeros(waves.shape)
        i = 0
        fmt = "{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}"
        print(fmt.format('New Wave', 'New T', 'W1', 'W2', 'T1', 'T2'))
        for j, new_wave in enumerate(waves):
            while waves_all[i] < new_wave:
                i += 1
            flux_new[j] = np.interp(new_wave, waves_all[i - 1:i + 1], flux_all[i - 1:i + 1])
            i += 1
        return flux_new, flux_units
