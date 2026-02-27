#!/usr/bin/env python
"""

"""
import math
import time
import numpy as np
import scipy.signal
# from astropy.units import *
from astropy import units as u
from astropy.io.fits import ImageHDU
from lmsdist_util import Util
from lms_globals import Globals
from lms_filer import Filer
from lms_detector import Detector
from lmssim_model import Model
from lmsdist_polyfit import PolyFit


class Toy:

    def __init__(self):
        return

    @staticmethod
    def run(sim_config):
        """ Trial new version which creates an EFP cube, then transforms it onto the detectors.
        """

        analysis_type = 'distortion'
        coord_in = 'efp_x', 'efp_y', 'wavelength'
        coord_out = 'det_x', 'det_y'

        nom_date_stamp = '20240109'
        nom_config = (analysis_type, Globals.nominal, nom_date_stamp,
                      'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
                      coord_in, coord_out)

        spifu_date_stamp = '20260112'
        spifu_config = (analysis_type, Globals.extended, spifu_date_stamp,
                        'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                        coord_in, coord_out)
        model_configurations = {Globals.nominal: nom_config, Globals.extended: spifu_config}

        model = Model()
        print(model)

        # =================================================================================================
        t_start = time.perf_counter()

        # Model PSFs are available up to 4 slices away from the slice the target is centred on.
        zemax_psf_slice_range = [-4, 4]

        for obs_name in sim_config:

            efp_w_row = None            # Initialise variables which may be assigned within loops etc.
            obs_cfg = sim_config[obs_name]
            wave_cen = float(obs_cfg['wave_cen'])*u.nm
            opticon = obs_cfg['opticon']

            lms_pp1 = obs_cfg['lms_pp1']
            model_config = model_configurations[opticon]
            PolyFit(opticon)                    # Instantiate the polynomial fit tools.
            filer = Filer()
            filer.set_configuration(analysis_type, opticon)

            # Calculate the prism and grating angles required to observe the target wavelength.
            # We use the nominal mode prism > wavelength calibration for all cases.
            filer.set_configuration(analysis_type, opticon)
            wpa_fit, wxo_fit, term_fits = filer.read_fit_parameters(opticon)
            lms_cfg = PolyFit.wave_to_config(wave_cen.value / 1000., opticon, wpa_fit, wxo_fit, select='min_ech_ang')
            # fit_slice_transforms = PolyFit.make_slice_transforms(lms_cfg, term_fits)
            date_stamp = model_config[2]

            # Read header and data shape (only) in from the template.
            hdu_list = filer.read_zemax_fits('../config/sim_template.fits') # , data_exts=data_exts)
            primary_header = hdu_list[0].header
            det_shape = hdu_list[1].data.shape
            _, n_det_cols = det_shape

            fp_mask = model.get_fp_mask(obs_cfg['wcu_mask'], obs_cfg['cfo_mask'])
            if fp_mask['id'] != 'open':
                fp_mask['slice_range'] = zemax_psf_slice_range
                efp_xy_list = fp_mask['efp_xy']
                fp_mask['slice_no_cen'] = []
                for efp_xy in efp_xy_list:
                    efp_x, efp_y = efp_xy
                    slice_no_cen, _ = Util.efp_y_to_slice(efp_y * u.mm)  # Pinhole is centred somewhere in this slice
                    fp_mask['slice_no_cen'].append(slice_no_cen)

            dit = float(obs_cfg['dit'])         # 1.3  # Integration time in seconds.
            ndit = int(obs_cfg['ndit'])         # No. of integrations

            # Find EFP bounds
            efp_xmax = Globals.efp_x_fov_mm
            xh = efp_xmax / 2.
            wrange = 2.*u.micron        # Width of wavelength range to model (4 um to overfill mosaic in spifu mode)
            wbounds = [wave_cen - wrange/2., wave_cen + wrange/2.]
            srp_model = 200000

            # Load selected extended background spectrum (units ph/s/m2/as2/um) for wavelength range which overfills mosaic
            # f_units_ext_in = 'phot/s/m2/um/arcsec2'
            w_ext, f_ext = model.get_flux(wbounds, obs_cfg['bgd_src'])
            f_pnh = np.zeros(f_ext.shape)
            if fp_mask['id'] != 'open':
                w_ext, f_ext = model.get_flux(wbounds, fp_mask['mask_ext'])
                w_pnh, f_pnh = model.get_flux(wbounds, obs_cfg['bgd_src'])
            is_dark = lms_pp1 == 'closed'   # Dark using LMS PP1
            if is_dark:
                f_ext *= 0.
                f_pnh *= 0.
            affines = filer.read_fits_affine_transform(date_stamp)
            svd_transforms = filer.read_svd_transforms(exc_tags=['fit_parameters', 'mfp_dfp'])
            # Find the list of closest svd transforms for each slice
            opt_transforms, ech_orders = Util.find_closest_transforms(wave_cen, opticon, svd_transforms)

            n_det_rows_slice = 250          # Number of rows per slice, with comfortable margin...
            image_mosaic, waves_mosaic, tau_ech_mosaic = [], [], []
            for det_no in range(1, 5):
                image_mosaic.append(np.zeros(det_shape))
                waves_mosaic.append(np.zeros(det_shape))
                tau_ech_mosaic.append(np.zeros(det_shape))

            # Set up dictionary of blaze wavelengths from ech_angle=0 transforms
            blaze = Model.make_blaze_dictionary(opt_transforms)
            out_folder = '../data/test_toysim/'

            print('Running simulation - ' + obs_name)
            fmt = "{:>10s},{:>6s},{:>6s},{:>6s},{:>8s},{:>8s},{:>8s},{:>8s},{:>15s},{:>11s}"
            title_txt = fmt.format('t_elapsed', 'det_no', 'slice', 'spifu', 'pri_ang', 'ech_ang', 'ech_ord',
                                   'w_blaze', 'w_range', 'det_rows')

            for tr_count, eo_slice_id in enumerate(list(opt_transforms)):

                if tr_count % 10 == 0:
                    print(title_txt)

                opt_transform = opt_transforms[eo_slice_id]
                slice_cfg = opt_transform.slice_configuration
                lms_cfg = opt_transform.lms_configuration
                ech_ord = slice_cfg['ech_ord']
                ech_ang = lms_cfg['ech_ang']
                pri_ang = lms_cfg['pri_ang']
                slice_no = slice_cfg['slice_no']
                spifu_no = slice_cfg['spifu_no']

                # txt += "pri_ang= {:5.3f}, ech_ang= {:5.3f}, ".format(pri_ang, ech_ang)
                w_blaze, tau_blaze = Model.make_tau_blaze(blaze, ech_ord, ech_ang)  # Make echelle blaze profile for this order.

                idx_max = np.argmax(tau_blaze)
                w_blaze_max = w_blaze[idx_max]

                # Load PSF library.  PSFs are sampled at 4x detector resolution so need to be down-sampled.
                # To start with, we just use PSFs for the boresight slice (slice no. 13)
                psf_dict = Model.load_psf_dict(opticon, ech_ord, downsample=True)
                _, psf_ext = psf_dict[0]
                w_min, w_max = slice_cfg['w_min']*u.micron, slice_cfg['w_max']*u.micron
                yc = Util.slice_to_efp_y(slice_no, 0.).value  # Slice y (beta) coordinate in EFP
                efp_y = np.array([yc, yc])
                efp_x = np.array([0., 0.])              # Slice x (alpha) bounds in EFP, map to dfp_y
                efp_w = np.array([w_min.value, w_max.value]) * u.micron
                efp_slice = {'efp_y': efp_y, 'efp_x': efp_x, 'efp_w': efp_w}    # Centre of slice for detector
                mfp_slice, oob = Util.efp_to_mfp(opt_transform, efp_slice)
                dfp_slice = Util.mfp_to_dfp(affines, mfp_slice)
                det_row_min = int(dfp_slice['dfp_y'][0] - 100)     # Bracket slices which typically cover 120 rows.
                det_row_max = det_row_min + n_det_rows_slice
                strip_shape = n_det_rows_slice, n_det_cols

                for det_no in dfp_slice['det_nos']:
                    det_idx = det_no - 1
                    ext_sig = np.zeros(strip_shape)
                    psf_sig = np.zeros(strip_shape)

                    dfp_det_nos = np.full(n_det_cols, det_no)
                    dfp_pix_xs = np.arange(n_det_cols)  # Detector column indices

                    n_rows_written = 0
                    w_illuminated = []
                    for det_row in range(det_row_min, det_row_max + 1):
                        strip_row = det_row - det_row_min
                        dfp_pix_ys = np.full(n_det_cols, det_row)
                        dfp_row = {'dfp_x': dfp_pix_xs, 'dfp_y': dfp_pix_ys, 'det_nos': dfp_det_nos}
                        efp_row = Util.dfp_to_efp(opt_transform, affines, dfp_row)
                        efp_x = efp_row['efp_x']
                        idx_illum = np.argwhere(np.abs(efp_x) < xh.value)
                        n_ib = len(idx_illum)
                        if n_ib == 0:                       # Skip unilluminated rows.
                            continue
                        # Apply extended spectrum (sky or black body) to illuminated rows
                        efp_w_row = efp_row['efp_w']*u.micron
                        w_obs = efp_w_row[idx_illum][:]
                        tau_echelle = np.interp(w_obs, w_blaze, tau_blaze)
                        tau_ech_mosaic[det_idx][det_row, idx_illum] = tau_echelle
                        f_ext_obs = np.interp(w_obs, w_ext, f_ext)
                        ext_sig[strip_row, idx_illum] = f_ext_obs * tau_echelle
                        waves_mosaic[det_idx][det_row, idx_illum] = w_obs
                        w_illuminated = w_illuminated + list(w_obs[:])
                        n_rows_written += 1
                    # Now convolve background flux map with bright slice psf. (ideally use filled slice psf)
                    image = image_mosaic[det_idx]
                    image[det_row_min:det_row_max, :] += scipy.signal.convolve2d(ext_sig, psf_ext, mode='same', boundary='symm')

                    if fp_mask['id'] in ['open', 'closed'] or lms_pp1 == 'closed':
                        continue

                    # Get dictionary of pinhole PSFs for all slices illuminated by a pinhole

                    slice_no_pnh_cens = np.array(fp_mask['slice_no_cen'])
                    slice_nos = np.full(slice_no_pnh_cens.shape, slice_no)
                    slice_no_offsets = slice_nos - slice_no_pnh_cens
                    sno_radius = 5 if opticon == Globals.nominal else 2
                    in_range_indices = np.argwhere(np.absolute(slice_no_offsets) < sno_radius)
                    if len(in_range_indices) == 0:
                        continue
                    psf_pnh_list = []
                    for idx in in_range_indices:
                        slice_no_offset = slice_no_offsets[idx][0]
                        _, psf_pnh = psf_dict[slice_no_offset]
                        psf_pnh_list.append(psf_pnh)

                    print("Applying pinhole spectra")
                    efp_xy_list = fp_mask['efp_xy']
                    for efp_xy in efp_xy_list:

                        efp_x_cfo = np.full(n_det_cols, efp_xy[0])
                        efp_y_cfo = np.full(n_det_cols, efp_xy[1])

                        efp_pnh = {'efp_x': efp_x_cfo, 'efp_y': efp_y_cfo, 'efp_w': efp_w_row}
                        dfp_pnh = Util.efp_to_dfp(opt_transform, affines, efp_pnh)
                        # Row by row population of psf_illum image
                        dfp_y_pnh = dfp_pnh['dfp_y']
                        dfp_rows_pnh = np.round(dfp_y_pnh).astype(int)
                        det_row_min_psf, det_row_max_psf = np.amin(dfp_rows_pnh), np.amax(dfp_rows_pnh)
                        for det_row in range(det_row_min_psf, det_row_max_psf + 1):
                            idx_illum = np.argwhere(dfp_rows_pnh == det_row)
                            n_ib = len(idx_illum)
                            if n_ib == 0:               # Skip unilluminated rows.
                                continue
                            # Apply extended spectrum (sky or black body) to illuminated rows
                            strip_row = det_row - det_row_min
                            w_obs = efp_w_row[idx_illum][:]
                            f_pnh_obs = np.interp(w_obs, w_pnh, f_pnh)
                            tau_echelle = np.interp(w_obs, w_blaze, tau_blaze)
                            psf_sig[strip_row, idx_illum] = f_pnh_obs * tau_echelle
                            image[det_row_min:det_row_max, :] += scipy.signal.convolve2d(psf_sig, psf_pnh,
                                                                                    mode='same', boundary='symm')
                t_now = time.perf_counter()
                t_el = int(t_now - t_start)

                fmt = "{:10d},{:6d},{:6d},{:6d},{:8.3f},{:8.3f},{:8d},{:8.0f},{:8.0f},{:6.0f},{:5d},{:5d}"
                txt = fmt.format(t_el, det_no, slice_no, spifu_no, pri_ang, ech_ang, ech_ord,
                                 int(w_blaze_max.to(u.nm).value), int(w_min.to(u.nm).value), int(w_max.to(u.nm).value),
                                 det_row_min, det_row_max)
                print(txt)

            n_obs = int(obs_cfg['nobs'])
            for obs_idx in range(0, n_obs):

                # Add dark current and read noise to illumination image (Finger, Rauscher)
                hdu_list = []
                for det_no in range(1, 5):
                    det_idx = det_no - 1
                    image = image_mosaic[det_idx]
                    frame = Detector.detect(image, dit, ndit)
                    hdu = ImageHDU(frame)
                    hdu.name = "DET{:d}.DATA".format(det_no)
                    hdu.header['ID'] = "{:d}".format(det_no)
                    c = 19.547
                    crval1d = {1: +c, 2: -c, 3: -c, 4: +c}
                    crval2d = {1: +c, 2: +c, 3: -c, 4: -c}
                    hdu.header['CRVAL1D'] = "{:8.3f}".format(crval1d[det_no])
                    hdu.header['CRVAL2D'] = "{:8.3f}".format(crval2d[det_no])
                    hdu.header['X_CEN'] = "{:8.3f}".format(crval1d[det_no])
                    hdu.header['Y_CEN'] = "{:8.3f}".format(crval2d[det_no])
                    el_adu = 2.0
                    hdu.header['HIERARCH ESO DET3 CHIP GAIN'] = "{:8.2f}".format(el_adu)
                    hdu_list.append(hdu)

                print()
                obs_tag = "_{:03d}.fits".format(obs_idx)
                for key in lms_cfg:
                    primary_header['AIT ' + key.upper()] = lms_cfg[key]
                for key in slice_cfg:
                    primary_header['AIT ' + key.upper()] = slice_cfg[key]

                file_name = obs_name + obs_tag
                print("Writing fits file - {:s}{:s}".format(out_folder, file_name))
                mosaic = file_name, primary_header, hdu_list
                filer.write_mosaic(out_folder, mosaic)

            debug = False
            if debug:
                fits_waves_out_path = out_folder + '_waves.fits'
                print("Writing fits file - {:s}".format(fits_waves_out_path))
                mosaic = file_name + '_waves', primary_header, hdu_list
                filer.write_mosaic(out_folder, mosaic)

                fits_tau_ech_out_path = out_folder + '_tau_ech.fits'
                mosaic = file_name + '_tau_ech', primary_header, hdu_list
                filer.write_mosaic(out_folder, mosaic)
        return
