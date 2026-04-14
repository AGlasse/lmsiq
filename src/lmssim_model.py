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
from synphot.models import BlackBody1D
from synphot import SourceSpectrum, units as s_units


class Model:

    tau_blaze_kernel = None # Kernel blaze profile tau(x) where x = (wave / blaze_wave(eo) - 1)
    tau_sky_cfo = 0.7       # Sky to CFO pinhole mask
    tau_wcu_cfo = 0.1       # WCU hot source and laser outputs to CFO pinhole mask
    tau_wfp_cfo = 0.9       # WCU output focal plane to CFO pinhole mask
    tau_cfo_lms = 0.2       # CFO pinhole mask to LMS detector (excluding detector qe and echelle blaze profile)
    tau_lms_ext = 0.8       # Transmission through extended mode optics.

    tau_sky = tau_sky_cfo * tau_wfp_cfo * tau_cfo_lms       # Leiden sky to detectors
    tau_whs = tau_wcu_cfo * tau_wfp_cfo * tau_cfo_lms       # WCU integrating sphere input to detectors
    tau_wfp = tau_whs / tau_wcu_cfo                         # WCU exit focal plane mask to detectors

    # Define a list of extended illumination sources.  These images will be convolved with the 'target slice' PSF.
    bgd_srcs = {'dark': {'sed': 'dark'},
                'wcu_bb': {'sed': 'bb', 'temperature': 1000., 'tau': tau_whs},  # 1000 K black body
                'cfo_mask': {'sed': 'bb', 'temperature': 70., 'tau': tau_cfo_lms},
                'wcu_mask': {'sed': 'bb', 'temperature': 300., 'tau': tau_wfp},
                'wcu_ls': {'sed': 'laser', 'power': 5.E+09, 'wavelength': 3.390, 'wrange':0, 'tau': tau_whs},
                'wcu_ll': {'sed': 'laser', 'power': 5.E+09, 'wavelength': 5.240, 'wrange':0, 'tau': tau_whs},
                'wcu_lt': {'sed': 'laser', 'power': 5.E+09, 'wavelength': 4.700, 'wrange':100, 'tau': tau_whs},
                'sky': {'sed': 'sky', 'tau': tau_sky}  # Model sky emission spectrum
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
        if wmin.unit != wmax.unit:
            wmax = wmax.to(wmin.unit)
        waves = np.arange(wmin.value, wmax.value, delta_w.value)*wmin.unit
        return waves

    def get_flux(self, wbounds, src, lt_w_offset):
        """ Calculate selected extended background spectrum (units el/s/pixel) for a wavelength range which
        overfills the instantaneous spectral coverage. Output units should be photons/pixel/second
        """
        w_ext = self._make_waves(wbounds)
        f_ext = None
        source = self.bgd_srcs[src]
        srp = 100000
        sed = source['sed']
        sample_etendue = Globals.elt_area.to(u.cm2) * Globals.alpha_pix * Globals.beta_slice  # AOmega cm^2 mas^2
        pixel_delta_w = w_ext / srp / Globals.pix_spec_res_el
        if sed == 'bb':
            f_bb = Model.black_body(w_ext, tbb=source['temperature'])  # Units are ph sec-1 micron-1 cm-2 mas-2
            pixel_delta_w = w_ext / srp / Globals.pix_spec_res_el
            f_ext = sample_etendue * pixel_delta_w.to(u.micron) * Model.tau_whs * f_bb      # ph / sec / pix
        if sed == 'sky':
            f_sky = Model.load_sky_emission(w_ext)      # Units = ph/s/m2/um/arcsec2
            sample_etendue = Globals.elt_area.to(u.m2) * Globals.alpha_pix.to(u.arcsec) * Globals.beta_slice.to(u.arcsec)
            f_ext = sample_etendue * pixel_delta_w.to(u.micron) * Model.tau_sky * f_sky
        if sed == 'laser':
            f_laser = Model.build_laser_emission(source, w_ext, lt_w_offset)
            f_ext_in = Model.tau_whs * f_laser
            atel = math.pi * (39. / 2)**2 *u.m *u.m     # ELT collecting area
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
        for transform in transforms:
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
    def build_laser_emission(laser, waves, wave_offset):
        """ Calculate laser signal as a spectrum with units photlam = photon/sec/cm2/angstrom/sterad
        Assume 10 mW total laser output over-filling the METIS
        field of view by a factor of 2, taken as A = pi x a x a where a = 6 arcsec x sqrt(2), giving A = 72 x 3.14 = 230 arcsec^2
        Attenuation by the integrating sphere and optics is set = 1.
        :param waves:
        :param laser:
        :param wave_offset:
        :return: flux quantity in units ph/s
        """
        laser_wave = (laser['wavelength'] + wave_offset)*u.micron
        laser_power = laser['power']
        idx_cen = np.argwhere(waves - laser_wave < 0)[:, 0][-1]
        line_width = waves[idx_cen] / 100000
        pix_fwhm = line_width / (waves[idx_cen] - waves[idx_cen-1])
        pix_sigma = pix_fwhm / 2.355
        pix_hw = 5
        n_pix_hw = 2 * pix_hw + 1
        indices = np.arange(n_pix_hw)        # 101 pixel scale, line centred at pixel 50.
        lsf = Globals.gauss(indices, laser_power, 5., pix_sigma)
        laser_flux = np.zeros(waves.shape)
        laser_flux[idx_cen - pix_hw: idx_cen + pix_hw + 1] = lsf
################################################ TEMPORARY KLUDGE
##        laser_flux = 1.E8 * np.sin(waves * u.rad * 10000. / u.micron) ** 2
################################################
        return laser_flux

    @staticmethod
    def black_body(waves, tbb=1000.):
        """ Generate black body emission spectrum using the SynPhot BlackBody1D model.  The output has been checked
        against Mathcad to to have the documented units of (ph/sec/Angstrom/cm2/steradian).
        """
        bb = SourceSpectrum(BlackBody1D, temperature=tbb * u.K)
        angstrom_micron = 1.0E4
        k = angstrom_micron / Globals.mas2_sterad
        flux_photlam = bb(waves.to(u.micron))
        flux = flux_photlam.value * k * u.ph / u.cm / u.cm / u.mas / u.mas / u.s / u.micron
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
