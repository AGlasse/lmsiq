#!/usr/bin/env python
"""

"""
import math
import numpy as np
import synphot.units
from astropy.io import fits
from astropy import units as u, constants as const
from lmsdist_util import Util
from lms_globals import Globals
from lms_filer import Filer
from lms_detector import Detector
from synphot.models import BlackBody1D
from synphot import SourceSpectrum, units as s_units


class Model:

    tau_blaze_kernel = None  # Kernel blaze profile tau(x) where x = (wave / blaze_wave(eo) - 1)
    tau_cfo_mask = 1.0      # CFO pinhole mask to detector (excluding detector and echelle blaze profile)
    tau_wcu_fp = .9         # WCU focal plane to METIS window
    tau_wcu_hs = .05         # WCU hot source to METIS window
    tau_sky = .5            # Sky to METIS window
    tau_lms_toy = .2        # Factor to use in the toy simulator only (assume ScopeSim covered)

    # Define a list of extended illumination sources.  These images will be convolved with the 'target slice' PSF.
    bgd_srcs = {'dark': {'sed': 'dark'},
                'wcu_bb': {'sed': 'bb', 'temperature': 1000., 'tau': tau_wcu_hs},               # 1000 K black body
                'cfo_mask': {'sed': 'bb', 'temperature': 70., 'tau': tau_cfo_mask},
                'wcu_mask': {'sed': 'bb', 'temperature': 300., 'tau': tau_wcu_fp},
                'wcu_ls': {'sed': 'laser', 'flux': 1.E+09, 'wavelength': 3390, 'wrange':0, 'tau': tau_wcu_hs},
                'wcu_ll': {'sed': 'laser', 'flux': 1.E+09, 'wavelength': 5240, 'wrange':0, 'tau': tau_wcu_hs},
                'wcu_lt': {'sed': 'laser', 'flux': 1.E+09, 'wavelength': 4700, 'wrange':100, 'tau': tau_wcu_hs},
                'sky': {'sed': 'sky', 'tau': tau_sky}    # Model sky emission spectrum
                }

    # Define one or more (point-like) pinhole masks which will spatially filter the extended source.  The model
    # specified PSFs at +-4 slices from the target slice will be convolved with the 'pinhole' images.
    fp_masks = {'cfopnh': {'id': 'cfo', 'efp_xy': [[0., 0.]],           # On-axis pinhole in boresight
                        'mask_ext': 'cfo_mask'},
                'pinhole_lm': {'id': 'wcu', 'efp_xy': [[0., 0.]],       # Steerable pinhole in WCU.
                        'mask_ext': 'wcu_mask'},
                'grid_lm': {'id': 'wcu', 'efp_xy': [],           # Steerable pinhole in WCU.
                        'mask_ext': 'wcu_mask'},
                'open': {'id': 'open', 'efp_xy_cfo': None,     # FP-1 open position
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
        waves = np.arange(wmin.value, wmax.value, delta_w.value) * u.nm
        return waves

    def get_flux(self, wbounds, src):
        """ Load selected extended background spectrum (units ph/s/m2/as2/um) for wavelength range which overfills mosaic
        f_units_ext = 'phot/s/m2/um/arcsec2'
        """
        w_ext = self._make_waves(wbounds)
        f_ext = None
        source = self.bgd_srcs[src]
        sed = source['sed']
        tau_qe = source['tau'] * Detector.qe
        pixel_etendue = Globals.elt_area * Globals.alpha_pix * Globals.beta_slice
        if sed == 'bb':
            f_bb = Model.black_body(w_ext, tbb=source['temperature'])  # Units are photlam = ph/sec/cm2/angstrom/sterad
            # if simulator is Globals.scopesim:
            #     return w_ext, f_bb
            srp = 100000
            pixel_delta_w = w_ext / srp / Globals.pix_spec_res_el
            tau = Model.tau_wcu_hs * Model.tau_lms_toy
            f_ext = pixel_etendue.to(u.cm2 * u.sr) * pixel_delta_w.to(u.angstrom) * tau * f_bb
        if sed == 'sky':
            f_sky = Model.load_sky_emission(w_ext)      # Units = ph/s/m2/um/arcsec2
            # if simulator is Globals.scopesim:
            #     f_sky_scope = f_sky.to(u.plam)
            #     return w_ext, f_sky_scope

            f_ext_in = Model.tau_sky * Model.tau_lms_toy * f_sky
            atel = math.pi * (39. / 2)**2 *u.m *u.m               # ELT collecting area
            alpha_pix = Globals.alpha_pix               # Along slice pixel scale
            beta_slice = Globals.beta_slice             # Slice width
            delta_w = wbounds[0] / 100000               # Spectral resolution
            pix_delta_w = 2.5                           # Pixels per spectral resolution element
            f_ext = f_ext_in * atel * alpha_pix * beta_slice * delta_w / pix_delta_w
        if sed == 'laser':
            f_laser1 = Model.build_laser_emission(w_ext, source)          # photons/second/cm2/ang/sr
            # u.cm2sr = u.cm * u.cm * u.rad * u.rad
            # f_laser2 = f_laser1 * pixel_etendue.to(u.cm2sr)                           # ph/sec/pixel
            # f_laser3 = f_laser2 * w_ext.to(u.angstrom) / 100000
            f_ext_in = Model.tau_wcu_hs * Model.tau_lms_toy * f_laser1
            atel = math.pi * (39. / 2)**2 *u.m *u.m               # ELT collecting area
            alpha_pix = Globals.alpha_pix               # Along slice pixel scale
            beta_slice = Globals.beta_slice             # Slice width
            delta_w = wbounds[0] / 100000               # Spectral resolution
            pix_delta_w = 2.5                           # Pixels per spectral resolution element
            f_ext = f_ext_in * atel * alpha_pix * beta_slice * delta_w / pix_delta_w

        f_ext_min, f_ext_max = np.amin(f_ext), np.amax(f_ext)
        fmt = "Adding extended {:s} flux with min/max signal = {:10.1f}/{:10.1f} el/pix/sec"
        fmt.format(sed, f_ext_min, f_ext_max)
        return w_ext, f_ext

    def get_fp_mask(self, wcu_mask, cfo_mask):
        if cfo_mask == 'pnh':
            fp_mask = Model.fp_masks['cfopnh']
            return fp_mask
        if wcu_mask == 'open':
            fp_mask = Model.fp_masks['open']
            return fp_mask
        fp_mask = Model.fp_masks[wcu_mask]
        file_name = 'fp_mask_' + wcu_mask
        fp_mask['efp_xy'] = Filer.read_pinholes(file_name, xy_filter=(0.5, 1.0))
        return fp_mask

    @staticmethod
    def load_psf_dict(opticon, ech_ord, downsample=False, slice_no_tgt=13):
        analysis_type = 'iq'

        nominal = Globals.nominal
        nom_iq_date_stamp = '2024073000'
        nom_config = (analysis_type, nominal, nom_iq_date_stamp,
                      'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
                      None, None)

        spifu = Globals.extended
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
                hdu_list = filer.read_zemax_fits(file_path)
                hdr, psf = hdu_list[0].header, hdu_list[0].data

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

    @staticmethod
    def make_blaze_dictionary(transforms):
        blaze = {}
        for key in transforms:
            transform = transforms[key]
            lms_cfg = transform.lms_configuration
            slice_cfg = transform.slice_configuration
            if slice_cfg['slice_no'] != 13:
                continue
            ech_ang = lms_cfg['ech_ang']
            mfp_bs = {'mfp_x': [0.], 'mfp_y': [0.]}
            ech_ord = slice_cfg['ech_ord']
            efp_bs = Util.mfp_to_efp(transform, mfp_bs)
            wave = efp_bs['efp_w'][0]
            if ech_ord not in blaze:
                blaze[ech_ord] = {}
            blaze[ech_ord][ech_ang] = wave
        return blaze

    @staticmethod
    def build_laser_emission(waves, laser, wave_offset=0.*u.nm):
        """ Calculate laser signal as a spectrum with units photlam = photon/sec/cm2/angstrom/sterad
        Assume 10 mW total laser output over-filling the METIS
        field of view by a factor of 2, taken as A = pi x a x a where a = 6 arcsec x sqrt(2), giving A = 72 x 3.14 = 230 arcsec^2
        Attenuation by the integrating sphere and optics is set = 1.
        :param waves:
        :param laser:
        :param wave_offset:
        :return: flux quantity in units ph/s
        """
        laser_power = 1.e5
        idx_cen = np.argwhere(waves - 4700 * u.nm < 0)[:, 0][-1]
        line_width = waves[idx_cen] / 100000
        pix_fwhm = line_width / (waves[idx_cen] - waves[idx_cen-1])
        pix_sigma = pix_fwhm / 2.355
        indices = np.arange(11.)        # 101 pixel scale, line centred at pixel 50.
        lsf = Globals.gauss(indices, laser_power, 5., pix_sigma)
        laser_flux = np.zeros(waves.shape)
        laser_flux[idx_cen - 5: idx_cen + 6] = lsf
        return laser_flux

    @staticmethod
    def black_body(waves, tbb=1000.):
        """ Generate black body emission spectrum using the
        :param waves:
        :param tbb:
        :return:
         Blackbody docs say it returns units are PHOTLAM (ph/sec/Angstrom/cm2/steradian)
        """
        bb = SourceSpectrum(BlackBody1D, temperature=tbb * u.K)
        flux = bb(waves.to(u.angstrom))
        return flux

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
        return waves*u.micron, tk

    @staticmethod
    def load_psf_dict(opticon, ech_ord, downsample=False, slice_no_tgt=13):
        analysis_type = 'iq'

        nominal = Globals.nominal
        nom_iq_date_stamp = '2024073000'
        nom_config = (analysis_type, nominal, nom_iq_date_stamp,
                      'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
                      None, None)

        spifu = Globals.extended
        spifu_date_stamp = '2024061802'
        spifu_config = (analysis_type, spifu, spifu_date_stamp,
                        'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                        None, None)

        model_configurations = {nominal: nom_config, spifu: spifu_config}
        model_config = model_configurations[opticon]
        filer = Filer()
        filer.set_configuration('distortion', opticon)
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
                hdu_list = filer.read_zemax_fits(file_path)
                hdr, psf = hdu_list[0].header, hdu_list[0].data

                # print("slice_no={:d}, psf_max={:10.3e}".format(slice_no, np.amax(psf)))
                if downsample:
                    oversampling = 4
                    n_psf_rows, n_psf_ncols = psf.shape
                    n_ds_rows, n_ds_cols = int(n_psf_rows / oversampling), int(n_psf_ncols / oversampling)
                    psf = psf.reshape(n_ds_rows, oversampling, n_ds_cols, -1).mean(axis=3).mean(axis=1)  # down sample
                slice_no_offset = slice_no - slice_no_tgt
                psf_dict[slice_no_offset] = hdr, psf
                psf_sum += np.sum(psf)
            # Normalise the PSFs so that the total flux of all slices sums to unity in detector space
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
        waves_all = data_table['lam'] * u.nm
        flux_all, flux_errs = data_table['flux'], None
        # flux_units = u.ph / u.second / u.m / u.m / u.micron / u.arcsec / u.arcsec    # 'ph/s/m2/um/arcsec2'

        # print("Loaded sky transmission and emission spectrum with units {:s}".format(flux_units))
        flux = np.zeros(waves.shape)
        i = 0
        fmt = "{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}"
        print(fmt.format('New Wave', 'New T', 'W1', 'W2', 'T1', 'T2'))
        n_waves_all, = waves_all.shape
        #  Bug!  Need to implement long wavelength test for when requested wavelength range overruns sky spectrum..
        for j, new_wave in enumerate(waves):
            # print(i, waves_all[i], new_wave)
            while waves_all[i] <= new_wave:
                i += 1
            flux[j] = np.interp(new_wave, waves_all[i - 1:i + 1], flux_all[i - 1:i + 1])
            i += 1
            if i >= n_waves_all - 1:
                break
        return flux * synphot.units.PHOTLAM
