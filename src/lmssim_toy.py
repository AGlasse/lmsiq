#!/usr/bin/env python
"""

"""
import math
import time
import numpy as np
import scipy.signal
from astropy import units as u
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
        analysis_type = 'distortion'
        coord_in = 'efp_x', 'efp_y', 'wavelength'
        coord_out = 'det_x', 'det_y'

        nom_date_stamp = '20240109'
        nom_config = (analysis_type, Globals.nominal, nom_date_stamp,
                      'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
                      coord_in, coord_out)

        spifu_date_stamp = '20250110'
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
            obs_cfg = sim_config[obs_name]
            wave_cen = float(obs_cfg['wave_cen'])*u.nm
            opticon = Globals.nominal if obs_cfg['opticon'] == 'nominal' else Globals.extended

            lms_pp1 = obs_cfg['lms_pp1']
            model_config = model_configurations[opticon]
            PolyFit(opticon)                    # Instantiate the polynomial fit tools.
            filer = Filer(model_config)

            # Calculate the prism and grating angles required to observe the target wavelength.
            # We use the nominal mode prism > wavelength calibration for all cases.
            nom_config = model_configurations[Globals.nominal]
            nom_filer = Filer(nom_config)
            wpa_fit, _, _ = nom_filer.read_fit_parameters(Globals.nominal)
            _, wxo_fit, term_fits = filer.read_fit_parameters(opticon)
            lms_cfg = PolyFit.wave_to_config(wave_cen.value / 1000., wpa_fit, wxo_fit, select='min_ech_ang')
            fit_slice_transforms = PolyFit.make_slice_transforms(lms_cfg, term_fits)


            date_stamp = model_config[2]

            # Read header and data shape (only) in from the template.
            data_exts = [1, 2, 3, 4]
            hdr_primary, data_list = filer.read_zemax_fits('../config/sim_template.fits', data_exts=data_exts)
            det_shape = data_list[0].shape
            _, n_det_cols = det_shape

            fp_mask = model.get_fp_mask(obs_cfg['fp_mask'])
            if fp_mask['id'] != 'open':
                fp_mask['slice_range'] = zemax_psf_slice_range
                efp_y_cfo = fp_mask['efp_y_cfo'] * u.mm
                slice_no_cen, _ = Util.efp_y_to_slice(efp_y_cfo)  # Pinhole is centred on this slice
                fp_mask['slice_no_cen'] = slice_no_cen
                efp_x_cfo = fp_mask['efp_x_cfo'] * u.mm

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
            w_ext, f_ext = model.get_flux(Globals.toysim, wbounds, obs_cfg['bgd_src'])
            f_pnh = np.zeros(f_ext.shape)
            if fp_mask['id'] != 'open':
                w_ext, f_ext = model.get_flux(Globals.toysim, wbounds, fp_mask['mask_ext'])
                w_pnh, f_pnh = model.get_flux(Globals.toysim, wbounds, obs_cfg['bgd_src'])
            is_dark = lms_pp1 == 'closed'   # Dark using LMS PP1
            if is_dark:
                f_ext *= 0.
                f_pnh *= 0.
            affines = filer.read_fits_affine_transform(date_stamp)
            svd_transforms = filer.read_svd_transforms(exc_tags=['fit_parameters',
                                                                 'mfp_dfp']
                                                       )
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

            print('Running simulation - ' + obs_name)
            fmt = "{:>10s},{:>4s},{:>6s},{:>6s},{:>8s},{:>8s},{:>8s},{:>8s},{:>15s},{:>11s}"
            title_txt = fmt.format('t_elapsed', 'cfg', 'slice', 'spifu', 'pri_ang', 'ech_ang', 'ech_ord',
                                   'w_blaze', 'w_range', 'det_rows')

            for tr_count, eo_slice_id in enumerate(list(opt_transforms)):

                if tr_count % 10 == 0:
                    print(title_txt)

                opt_transform = opt_transforms[eo_slice_id]
                cfg = opt_transform.configuration
                cfg_id = cfg['cfg_id']
                ech_ord = cfg['ech_order']
                ech_ang = cfg['ech_ang']
                pri_ang = cfg['pri_ang']
                slice_no = cfg['slice_no']
                spifu_no = cfg['spifu_no']

                # txt += "pri_ang= {:5.3f}, ech_ang= {:5.3f}, ".format(pri_ang, ech_ang)
                w_blaze, tau_blaze = Model.make_tau_blaze(blaze, ech_ord, ech_ang)  # Make echelle blaze profile for this order.

                idx_max = np.argmax(tau_blaze)
                w_blaze_max = w_blaze[idx_max]

                # Load PSF library.  PSFs are sampled at 4x detector resolution so need to be down-sampled.
                # To start with, we just use PSFs for the boresight slice (slice no. 13)
                psf_dict = Model.load_psf_dict(opticon, ech_ord, downsample=True)
                _, psf_ext = psf_dict[0]
                w_min, w_max = cfg['w_min']*u.micron, cfg['w_max']*u.micron
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
                        # print('toy190 ', strip_row)
                        ext_sig[strip_row, idx_illum] = f_ext_obs * tau_echelle
                        waves_mosaic[det_idx][det_row, idx_illum] = w_obs
                        w_illuminated = w_illuminated + list(w_obs[:])
                        n_rows_written += 1
                    # w_ill_np = np.array(w_illuminated)
                    # w_ext_min, w_ext_max = np.nanmin(w_ill_np), np.nanmax(w_ill_np)

                    # Now convolve background flux map with bright slice psf. (ideally use filled slice psf)
                    image = image_mosaic[det_idx]
                    image[det_row_min:det_row_max, :] += scipy.signal.convolve2d(ext_sig, psf_ext, mode='same', boundary='symm')

                    if fp_mask['id'] in ['open', 'closed'] or lms_pp1 == 'closed':
                        continue

                    # Get dictionary of pinhole PSFs for all slices illuminated by a pinhole
                    slice_no_pnh_cen = fp_mask['slice_no_cen']
                    slice_no_offset = slice_no - slice_no_pnh_cen
                    sno_radius = 5 if opticon == Globals.nominal else 2
                    if math.fabs(slice_no_offset) < sno_radius:     # Pinhole PSF exists for this slice.
                        _, psf_pnh = psf_dict[slice_no_offset.value]
                    else:
                        continue

                    print("Applying pinhole spectrum")

                    efp_x_cfo = np.full(n_det_cols, fp_mask['efp_x_cfo'])
                    efp_y_cfo = np.full(n_det_cols, fp_mask['efp_y_cfo'])
                    efp_w_pnh = efp_w_row

                    efp_pnh = {'efp_x': efp_x_cfo, 'efp_y': efp_y_cfo, 'efp_w': efp_w_pnh}
                    dfp_pnh = Util.efp_to_dfp(opt_transform, affines, det_no, efp_pnh)
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

                fmt = "{:10d},{:4d},{:6d},{:6d},{:8.3f},{:8.3f},{:8d},{:8.0f},{:8.0f},{:6.0f},{:5d},{:5d}"
                txt = fmt.format(t_el, cfg_id, slice_no, spifu_no, pri_ang, ech_ang, ech_ord,
                                 int(w_blaze_max.to(u.nm).value), int(w_min.to(u.nm).value), int(w_max.to(u.nm).value),
                                 det_row_min, det_row_max)
                print(txt)

            n_obs = int(obs_cfg['nobs'])
            for obs_idx in range(0, n_obs):

                # Add dark current and read noise to illumination image (Finger, Rauscher)
                frame_mosaic = []
                for det_no in range(1, 5):
                    det_idx = det_no - 1
                    image = image_mosaic[det_idx]
                    frame = Detector.detect(image, dit, ndit)
                    frame_mosaic.append(frame)

                print()
                obs_tag = "_{:03d}.fits".format(obs_idx)
                toysim_out_path = '../data/test_toysim/' + obs_name + obs_tag
                print("Writing fits file - {:s}".format(toysim_out_path))
                for key in cfg:
                    hdr_primary['AIT ' + key.upper()] = cfg[key]

                filer.write_zemax_fits(toysim_out_path, hdr_primary, frame_mosaic)

            debug = False
            if debug:
                fits_out_path = toysim_out_path + '_illumination.fits'
                print("Writing fits file - {:s}".format(fits_out_path))
                filer.write_zemax_fits(fits_out_path, hdr_primary, image_mosaic)

                fits_waves_out_path = toysim_out_path + '_waves.fits'
                print("Writing fits file - {:s}".format(fits_waves_out_path))
                filer.write_zemax_fits(fits_waves_out_path, hdr_primary, waves_mosaic)

                fits_tau_ech_out_path = toysim_out_path + '_tau_ech.fits'
                print("Writing fits file - {:s}".format(fits_tau_ech_out_path))
                filer.write_zemax_fits(fits_tau_ech_out_path, hdr_primary, tau_ech_mosaic)

        return
