#!/usr/bin/env python
"""
Decorators for use in all LMS projects.  Currently just includes @debug

@author: Alistair Glasse

Update:
"""
import math
import sys
import warnings

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

import lmsdist
from lmsdist_util import Util
from lms_globals import Globals
from lms_mosaic import Mosaic
from lmsaiv_opt_tools import OptTools
from lmsaiv_plot import Plot
from lms_filer import Filer
from lms_transform import Transform


class Opt02:


    def __init__(self):
        return

    @staticmethod
    def dist(title, as_built, **kwargs):
        overwrite_as_built_traces = kwargs.get('overwrite_as_built_traces', True)

        test_name = 'lms_opt_02'
        opticon = Globals.nominal

        misalign_detector = False
        find_iso_alphas = True
        find_iso_lambdas = True
        calculate_transforms = True
        test_transforms = True

        snr_cut_alpha = 1.0     # Cut level to detect multiple (at least 3) iso-alpha traces from a single PSF

        inc_tags = [test_name, opticon[0:3]]         # Tokens to identify image files.
        filer = Filer()
        filer.set_configuration('distortion', opticon, is_ait_data=True)

        flag_text = 'Analysing'
        if find_iso_alphas:
            flag_text += ' iso-alpha'
            if find_iso_lambdas:
                flag_text += ' and'
        if find_iso_lambdas:
            flag_text += ' iso-lambda'
        if not find_iso_alphas and not find_iso_lambdas:
            flag_text = 'Calculating trace intersections only '
        else:
            flag_text += ' traces'
        print('{:s}'.format(flag_text))
        print()

        slice_map = as_built["slice_map_{:s}".format(opticon)]
        if Globals.is_debug('medium'):
            Plot.mosaic(slice_map, title='Slice Map', cmap='hsv', mask=(0.0, 'black'))
        slice_row_dict = Opt02._build_slice_row_dict(slice_map)         # Dictionary of per slice row bounds

        # Set up a deliberately misaligned detector
        test_det_no, test_det_offset, test_img_rot_deg = None, None, None
        if misalign_detector:
            test_det_no = 2
            test_det_offset = 0, 0
            test_det_rot_deg = 0.0
            test_img_rot_deg = -test_det_rot_deg
            print('Deliberately misaligning detector')
            print('det_no =       {:d}'.format(test_det_no))
            print('det_offset =   {:d}, {:d} arcsec'.format(test_det_offset[0], test_det_offset[1]))
            print('det_rotation = {:5.3f} deg.'.format(test_det_rot_deg))
            print()

        # ---------- ISO-ALPHA -------------------
        # Start with data loading and background subtraction.  The iso-alphas use the WCU BB and the lm_pinhole
        # mask.  The background under the trace is then the emission spectrum of the mask (at 300 K).  We remove
        # this by moving the pinhole out of the LMS field using the CFO chopper.
        if find_iso_alphas:
            print('Derive detector rotation and offset differences from iso-alpha trace data. ')
            alpha_traces = []

            # Load background images (BB flat field)
            print('Background')
            flat_step = {Globals.nominal: 'step1_', Globals.extended: 'step38_'}[opticon]
            bgd_mosaics = Filer.read_mosaic_list(Filer.test_data_folder, inc_tags + ['flat', flat_step])
            bgd_mosaic = bgd_mosaics[0]
            if misalign_detector:
                bgd_mosaic = OptTools.transform_detector_image(bgd_mosaics[0],
                                                               det_no=test_det_no,
                                                               xy_pix=test_det_offset,
                                                               angle=test_img_rot_deg)
            if Globals.is_debug('medium'):
                Plot.mosaic(bgd_mosaic, title='WCU lm_pinhole mask background', cmap='hot')
            sig_file_list = Filer.get_file_list(Filer.test_data_folder, inc_tags=inc_tags + ['iso_alpha'])
            # sig_mosaics = Filer.read_mosaic_list(inc_tags + ['iso_alpha'])
            for sig_file in sig_file_list:
                print()
                sig_mosaic = Filer.read_mosaic(Filer.test_data_folder, sig_file)
                print("Processing mosaic file {:s}".format(sig_mosaic.name))
                if misalign_detector:
                    sig_mosaic = OptTools.transform_detector_image(sig_mosaic,
                                                                   det_no=test_det_no,
                                                                   xy_pix=test_det_offset,
                                                                   angle=test_img_rot_deg)
                # Extract iso-alpha traces for all slices (by setting snr_cut=0.)
                tra_mosaic = sig_mosaic.subtract(bgd_mosaic)
                if Globals.is_debug('low'):
                    Plot.mosaic(sig_mosaic, title='Signal')
                if Globals.is_debug('low'):
                    Plot.mosaic(tra_mosaic, title='Signal - Bgd')

                # We find the multiple alpha traces in adjacent slices by accepting all traces above snr which
                # is low enough to avoid fit problems in un-illuminated slices, but to include enough traces to
                # calculate the PSF across-slice / beta phase.
                psf_alpha_traces = Opt02._extract_det_traces(tra_mosaic, 'iso-alpha', slice_map,
                                                             snr_cut=1.)
                # ..then fit a gaussian to all solid detections to find the single trace for the PSF centroid.
                # The slice plus beta phase should ideally(!) match the WCU derived efp_y parameter.
                # The method clips any points which are displaced by more than 'sigma_clip' times the rms
                # displacement from the best fit and then recalculates the best fit.
                alpha_trace_pair = Opt02._find_beta_phase(psf_alpha_traces, opticon, snr_cut=1.)
                if Globals.is_debug('high'):
                    print()
                    print('PSF iso-alpha traces found')
                    fmt = "{:>12s},{:>12s},{:>12s},"
                    print(fmt.format('Det. no.', 'Slice no.', 'Beta phase'))
                    fmt = "{:12d},{:12d},{:12.3f},"
                    for alpha_trace in alpha_trace_pair:
                        det_no = alpha_trace['det_no']
                        slice_no = alpha_trace['slice_no']
                        beta_phase = alpha_trace['beta_phase']
                        print(fmt.format(det_no, slice_no, beta_phase))

                if Globals.is_debug('low'):
                    overlay = {'type': 'det_traces', 'trace_type': 'iso_alpha', 'data': alpha_trace_pair}
                    Plot.mosaic(sig_mosaic, title='Background subtracted', cmap='grey', overlay=overlay)
                alpha_traces = alpha_traces + alpha_trace_pair         # 1 alpha traces per detector

            # Calculate mean angle between alpha traces and the row direction.
            if Globals.is_debug('high'):
                fmt = "{:>10s},{:>10s},{:>12s},{:>20s},"
                print(fmt.format('Det', 'Slice', 'Row     ', 'Mean rotation dispersion'))
                print(fmt.format('no.', 'no.  ', 'fiducial', 'dispersion to row'))
                print(fmt.format('-  ', '-    ', '-       ', '[degrees]'))
                fmt = "{:>10d},{:>10d},{:>12.2f},{:>20.2f},"
            for alpha_trace in alpha_traces:
                det_no = alpha_trace['det_no']
                slice_no = alpha_trace['slice_no']
                popt = alpha_trace['popt']
                row_fid = alpha_trace['v_fid']
                col_samples = np.arange(0, 2048, 128)
                gradients = Globals.polynomial(col_samples, *popt, gradient=True)
                mean_gradient = np.mean(gradients)
                # Calculate the angle between polynomials at the mosaic centre.
                deg_rad = 180./math.pi
                mean_theta_disp = -deg_rad * mean_gradient
                # Calculate the angle between the alpha trace (dispersion) and the detector row at the centre column
                if Globals.is_debug('high'):
                    print(fmt.format(det_no, slice_no, row_fid, mean_theta_disp))
                alpha_trace['mean_theta_disp'] = mean_theta_disp

            if 'alpha_traces' in as_built.keys() and not overwrite_as_built_traces:
                alpha_traces_archive = as_built['alpha_traces']
                alpha_traces_archive += alpha_traces
            as_built['alpha_traces'] = alpha_traces
            Filer.write_pickle(Globals.as_built_file, as_built)

        # -----------------------
        # ISO-LAMBDA
        if find_iso_lambdas:
            print('2. Extract iso-lambda traces to measure the intra-detector gap and the line spread function.')
            print('   The gap calculation will assume that the laser lines are spaced according to a smooth polynomial.')

            raw_mosaics = Filer.read_mosaic_list(inc_tags + ['iso_lambda'])
            # Plot.mosaic(raw_mosaics[0])
            mosaics = OptTools.median_subtract(raw_mosaics)
            # Plot.mosaic(mosaics[0])

            print(0)
            lambda_traces = []

            for mosaic in mosaics:
                print()
                print("Processing mosaic file {:s}".format(mosaic.name))
                if misalign_detector:
                    mosaic = OptTools.transform_detector_image(mosaic,
                                                               det_no=test_det_no,
                                                               xy_pix=test_det_offset,
                                                               angle=test_img_rot_deg)
                lt_wave = mosaic.primary_hdr['HIERARCH ACHG LASER WAVE']
                if Globals.is_debug('low'):
                    Plot.mosaic(mosaic, title='laser_wavelength = ' + "{:10.3f}".format(lt_wave))
                # Find the gap between detectors as a function of row number.
                snr_cut = 10
                mos_lambda_traces = Opt02._extract_det_traces(mosaic, 'iso-lambda', slice_map, snr_cut,
                                                              lt_wave=lt_wave)
                if len(mos_lambda_traces) == 0:
                    continue
                if Globals.is_debug('low'):
                    overlay = {'type': 'det_traces', 'trace_type': 'iso_lambda', 'data': mos_lambda_traces}
                    row_fid = mos_lambda_traces[0]['v_fid']
                    bounds = 0, 2048, row_fid-100, row_fid+100
                    Plot.mosaic(mosaic, title='Background subtracted', cmap='grey',
                                bounds=bounds, overlay=overlay)

                lambda_traces = lambda_traces + mos_lambda_traces

            if 'lambda_traces' in as_built.keys() and not overwrite_as_built_traces:
                lambda_traces_archive = as_built['lambda_traces']
                lambda_traces_archive += lambda_traces
            as_built['lambda_traces'] = lambda_traces
            Filer.write_pickle(Globals.as_built_file, as_built)

        alpha_traces = as_built['alpha_traces']
        lambda_traces = as_built['lambda_traces']

        # Find the intra-detector gap from a list of column position v wavelength for tunable laser spectral lines.
        gap_data, det_thetas = Opt02._find_detector_gaps(lambda_traces)
        if Globals.is_debug('low'):
            Plot.gap_data(gap_data, det_thetas)

        # Generate baseline affine transforms to project detector pixel coords into the mosaic focal plane.
        # We can/ought to project the detector intersections into the mosaic focal plane using the baseline
        # Zemax derived affine transforms.  If these are discrepant due to detector misalignment, they can
        # be adjusted explicitly using 'affine term offsets' for each detector.
        affines = Transform.create_affines(theta_offsets=[0.]*4,
                                           x_mfp_org_offsets=[0.]*4, y_mfp_org_offsets=[0.]*4,
                                           x_scale_offsets=[0.]*4, y_scale_offsets=[0.]*4)

        # Find all LMS configurations
        lms_config_ids = []
        for alpha_trace in alpha_traces:
            id = alpha_trace['lms_config_id']
            if id not in lms_config_ids:
                lms_config_ids.append(id)
        for lms_config_id in lms_config_ids:
            Plot.det_traces(alpha_traces, lambda_traces, lms_config_id)

        dist_coord = Opt02._find_trace_intersections(alpha_traces, lambda_traces, affines)
        _ = Opt02._print_dist_coord(filer, dist_coord, to_csv=False)

        path = Opt02._print_dist_coord(filer, dist_coord, to_csv=True)
        print("Coordinates written to file {:s}".format(path))
        print()

        for lms_config_id in dist_coord:
            coords = dist_coord[lms_config_id]
            Plot.det_traces(alpha_traces, lambda_traces, lms_config_id)
            # Plot.det_traces(alpha_traces, lambda_traces, lms_config_id, coords=coords, bounds=(0, 600, 330, 430))

        as_built['affines'] = affines
        as_built['dist_coord'] = dist_coord

        for lms_config_id in dist_coord:
            inc_tags = [lms_config_id]
            is_ait_data = False
            filer.set_configuration('distortion', opticon, is_ait_data)
            zemax_transforms = filer.read_svd_transforms(inc_tags=inc_tags)

            is_ait_data = True
            filer.set_configuration('distortion', opticon, is_ait_data)
            ait_transforms = filer.read_svd_transforms(inc_tags=inc_tags)
            # As a measure of accuracy, calculate the fractional difference between matrix terms,
            matching_transforms = Util.find_matching_transforms(ait_transforms, zemax_transforms)
            for ait_transform, zmx_transform in matching_transforms:
                if Globals.is_debug('low'):
                    a_slice_no = ait_transform.slice_configuration['slice_no']
                    fmt = "Slice {:d} fractional difference in matrix terms (AIT - ZMX) / ZMX"
                    print(fmt.format(a_slice_no))
                ait_matrices = ait_transform.matrices
                zmx_matrices = zmx_transform.matrices
                for key in ait_matrices:
                    if Globals.is_debug('low'):
                        fmt = "Matrix {:s}"
                        print(fmt.format(key.upper()))
                    ait_matrix = ait_matrices[key]
                    zmx_matrix = zmx_matrices[key]
                    with warnings.catch_warnings(record=True):
                        f_mat = (ait_matrix - zmx_matrix) / zmx_matrix
                    nr, nc = ait_matrix.shape
                    for r in range(0, nr):
                        text = "{:>6s}{:4d}".format('Row', r)
                        for c in range(0, nc - r):
                            f = f_mat[r, c]
                            text += "{:15.3f}".format(f)
                        if Globals.is_debug('low'):
                            print(text)

        if calculate_transforms:
            lmsdist.run()

        if test_transforms:
            print()
            # Better yet, use the transforms to do a forward and back ray trace.
            beta_phase = 0.5
            for ait_transform, zmx_transform in matching_transforms:
                slice_config = ait_transform.slice_configuration
                slice_no = slice_config['slice_no']

                efp_x = 0.
                efp_y = Util.slice_to_efp_y(slice_no, beta_phase, opticon).value
                efp_w = 0.5 * (slice_config['w_min'] + slice_config['w_max'])
                efp_points = {'efp_x': np.array([efp_x]), 'efp_y': np.array([efp_y]), 'efp_w': np.array([efp_w])}
                mfp_pts_ait = Util.efp_to_mfp(ait_transform, efp_points)
                mfp_pts_zmx = Util.efp_to_mfp(zmx_transform, efp_points)
        return as_built

    @staticmethod
    def _build_slice_row_dict(slice_map):
        # _, _, hdus = slice_map
        slice_row_dict = {}
        for hdu in slice_map.hdu_list:
            image, hdr = hdu.data, hdu.header
            det_no = int(hdr['ID'])
            slice_row_dict[det_no] = {}
            slice_nos = np.unique(image)
            for slice_no in slice_nos:
                if slice_no == 0.:
                    continue
                indices = np.argwhere(image == slice_no)
                row_min = np.min(indices[:, 0])
                row_max = np.max(indices[:, 0])
                slice_row_dict[det_no][slice_no] = {'row_min': row_min, 'row_max': row_max}
        return slice_row_dict

    @staticmethod
    def _find_beta_phase(all_alpha_traces, opticon, snr_cut=1.):
        """ Find the beta phase, the offset in slices of a PSF from the slice centre, from a list of alpha traces for
        a sliced PSF image.  For the extended mode only 3 spatial slices (11, 12, 13) are available, with up to 3
        images per detector.
        :param all_alpha_traces:
        :return:
        """
        alpha_trace_list = []

        for det_no in range(1, 5):
            pslice_no_range = Globals.pslice_no_range[opticon]
            for pslice_no in pslice_no_range:
                signal_list, slice_no_list, spifu_alpha_traces = [], [], []
                is_detected = False
                for alpha_trace in all_alpha_traces:
                    if det_no != alpha_trace['det_no']:
                        continue
                    trace_slice_no = alpha_trace['slice_no']
                    trace_fslice_no, trace_pslice_no = Util.decode_slice_no(trace_slice_no)
                    # trace_pslice_no = alpha_trace['pslice_no'] // 100
                    if pslice_no != trace_pslice_no:
                        continue
                    # Select all slices with the same pupil slice number
                    signal_list.append(alpha_trace['signal'])
                    slice_no_list.append(trace_slice_no)
                    spifu_alpha_traces.append(alpha_trace)
                    snr = alpha_trace['snr']
                    is_detected = True if snr > snr_cut else is_detected

                if len(signal_list) == 0 or not is_detected:
                    continue
                signals, slice_nos = np.array(signal_list), np.array(slice_no_list)
                sig_max = np.amax(np.array(signals))
                idx_max = np.argmax(signals)
                slice_no_peak = slice_nos[idx_max]
                alpha_trace = spifu_alpha_traces[idx_max].copy()
                psf_alpha_sigma = 2.5
                p0_guess = [sig_max, slice_no_peak, psf_alpha_sigma]
                is_error = False
                try:
                    gopt, gcov = curve_fit(Globals.gauss, slice_nos, signals, p0=p0_guess, maxfev=5000)
                except (RuntimeError, TypeError, ValueError):
                    print("!! Error finding beta-phase for det {:d} !!".format(det_no))
                    is_error = True
                if not is_error:
                    beta_phase = gopt[1] - slice_no_peak + 0.5       # Phase = 0.5 for source centred in slice
                    alpha_trace['beta_phase'] = beta_phase
                    alpha_trace_list.append(alpha_trace)
        return alpha_trace_list

    @staticmethod
    def _find_detector_gaps(lambda_traces, order=2):
        """ Calculate the gap width in pixels between detectors separated in the dispersion direction.  It does this
        by fitting traces for the same slice in both short and long wave detectors, so the data must include
        sufficient laser lines (>= 2 for a linear fit) on both detectors.
        """
        gap_data = {}
        for lambda_trace in lambda_traces:
            slice_no = lambda_trace['slice_no']
            if slice_no not in gap_data:
                gap_data[slice_no] = {'lams_l': [], 'lams_r': [], 'cols_l': [], 'cols_r': [], 'col_gap': None}
            det_no = lambda_trace['det_no']
            v_fid = lambda_trace['v_fid']
            lam = lambda_trace['efp_w']
            lams_tag, cols_tag = 'lams_l' , 'cols_l'
            if det_no in [2, 4]:
                lams_tag, cols_tag = 'lams_r' , 'cols_r'
            gap_data[slice_no][lams_tag].append(lam)
            gap_data[slice_no][cols_tag].append(v_fid)
            gap_data[slice_no]['u_mean'] = lambda_trace['u_mean']
            gap_data[slice_no]['det_no'] = det_no

        for slice_no in gap_data:
            gap_sample = gap_data[slice_no]
            idx_l = np.argsort(np.array(gap_sample['lams_l']))
            idx_r = np.argsort(np.array(gap_sample['lams_r']))
            lams_l = np.array(gap_sample['lams_l'])[idx_l]
            lams_r = np.array(gap_sample['lams_r'])[idx_r]
            cols_l = np.array(gap_sample['cols_l'])[idx_l]
            cols_r = np.array(gap_sample['cols_r'])[idx_r]
            # with warnings.catch_warnings(record=True):
            if cols_l.shape[0] <= order or cols_r.shape[0] <= order:
                print('Insufficient iso-lambda traces for slice ', slice_no)
                continue
            lam_popt_l, lam_mid_l = None, None
            try:
                lam_popt_l, _ = curve_fit(Globals.polynomial, cols_l, lams_l, p0=[0.]*order)
            except:
                print('Failed curve_fit ')
            lam_mid_l = Globals.polynomial(2048., *lam_popt_l)[0]     # Wavelength of RH edge of left hand detector
            try:
                col_popt_r, _ = curve_fit(Globals.polynomial, lams_r, cols_r, p0=[0.]*order)
            except:
                print('Failed curve_fit ')
            col_gap = -1. * Globals.polynomial(lam_mid_l, *col_popt_r)[0]
            gap_sample['col_gap'] = col_gap

        # Find angle between 1,2 and 3,4 by linear fitting to the intra-detector gap as a function of row number.
        det_thetas = {}
        for det_nos in ['12', '34']:
            det_thetas[det_nos] = None
            col_gap_list, row_list = [], []
            for slice_no in gap_data:
                gap_sample = gap_data[slice_no]
                if gap_sample['col_gap'] is None:       # No gap value available
                    continue
                if str(gap_sample['det_no']) in det_nos:
                    col_gap_list.append(gap_sample['col_gap'])
                    row_list.append(gap_sample['u_mean'])
            if len(col_gap_list) == 0:
                print('Insufficient gap data for detector pair ', det_nos)
                continue

            rows = np.array(row_list)
            gaps = np.array(col_gap_list)
            mean_gap = np.mean(gaps)

            linear_guess = [mean_gap, 0.]
            try:
                print(rows, gaps)
                popt, pcov = curve_fit(Globals.polynomial, rows, gaps, p0=linear_guess)
            except OptimizeWarning:
                print('Warning: polynomial fit failed')
            gradient = Globals.polynomial(mean_gap, *popt, gradient=True)[0]
            gradient_err = np.sqrt(pcov[1][1])
            deg_rad = 180. / math.pi
            theta = math.atan(gradient) * deg_rad
            theta_err = gradient_err * deg_rad / (1 + gradient ** 2)
            det_thetas[det_nos] = theta, theta_err
        return gap_data, det_thetas

    @staticmethod
    def _extract_det_traces(mosaic, trace_type, slice_map, snr_cut,
                            lt_wave=None, sigma_clip=2.):
        """ Extract iso-alpha or iso-lambda traces from a spectral image.
        :param mosaic: Data tuple (name, _, image list)  Background subtracted trace mosaic
        :param trace_type: 'iso-alpha' or 'iso-lambda'
        :param slice_map:
        :param kwargs:
        :return:
        """
        name, prim_hdr, hdul = mosaic.name, mosaic.primary_hdr, mosaic.hdu_list

        is_alpha = trace_type == 'iso-alpha'

        efp_w = -1. if is_alpha else lt_wave
        efp_chop_x = prim_hdr['HIERARCH ACHG CFO CHOP X']
        efp_chop_y = prim_hdr['HIERARCH ACHG CFO CHOP Y']
        efp_x, efp_y = efp_chop_x, efp_chop_y

        ref_ech_ord = prim_hdr['HIERARCH ACHG REF_ECH_ORD']
        opticon = prim_hdr['HIERARCH AIT OPTICON']
        pri_ang = prim_hdr['HIERARCH AIT PRI_ANG']
        ech_ang = prim_hdr['HIERARCH AIT ECH_ANG']

        det_traces = []
        # _, _, slice_map_hdus = slice_map
        fmt = ''
        if Globals.is_debug('high'):
            rowcol_tag = 'Row' if is_alpha else 'Column'
            sli_tag = rowcol_tag + ' in slice'
            abs_tag = rowcol_tag + ' in image'
            print()
            print("Slice by slice {:s} trace extraction".format(trace_type))
            fmt = "{:>12s},{:>12s},{:>20s},{:>20s},{:>12s},{:>12s},{:>12s},{:>12s}"
            print(fmt.format('Detector', 'Slice ', 'Brightest', 'Brightest', 'Signal', 'Bgd', 'Noise', 'SNR'))
            print(fmt.format('Number  ', 'Number',  sli_tag   , abs_tag    ,   'DN/sec', 'DN/sec', '1 sigma', '-'))
            fmt = "{:>12d},{:>12d},{:>20d},{:>20d},{:>12.2f},{:>12.2f},{:>12.2f},{:>12.1f}"

        for hdu in hdul:
            det_no = int(hdu.header['ID'].strip())
            mos_idx = Globals.mos_idx[det_no]
            image = np.array(hdu.data)

            slice_map_data = np.array(slice_map.hdu_list[mos_idx].data)
            snu = np.unique(slice_map_data)
            slice_nos = [int(s) for s in snu if s > 0.]
            # Start by extracting the data for each slice.
            for slice_no in slice_nos:
                ism = int(0.5 * Globals.intra_slice_gap)                # Intra-slice margin
                idx = np.argwhere(slice_no == slice_map_data)
                slice_row_min, slice_row_max = np.amin(idx[:, 0]) - ism, np.amax(idx[:, 0]) + ism

                # Extract slice image
                slice_image = np.array(image[slice_row_min:slice_row_max, :])
                (u_axis, v_axis) = (1, 0) if is_alpha else (0, 1)
                v_count = slice_image.shape[v_axis]

                # We locate traces by collapsing along the u direction (rows for iso-alpha, columns for iso-lambda)
                # Currently only the brightest trace in each slice is found, so the input data should be for a
                # single pinhole only.  This approach simplifies the estimation of beta_phase, the across slice
                # position of the trace.
                noise = np.median(np.std(slice_image, axis=u_axis))      # Across-trace noise measure for slice image
                bgd = np.median(slice_image)

                s_aves = np.mean(slice_image, axis=u_axis)
                v_max_signal = np.argmax(s_aves)                         # Brightest row in slice (alpha) / column (lambda)
                row_max_signal = slice_row_min + v_max_signal
                v1_sig, v2_sig = v_max_signal - 5, v_max_signal + 5      # Rows/cols to use for trace analysis
                v1_sig, v2_sig = max(0, v1_sig), min(v_count, v2_sig)
                signal = np.mean(s_aves[v1_sig: v2_sig])
                snr = (signal - bgd) / noise
                is_low_snr = snr < snr_cut
                if Globals.is_debug('high'):
                    text = fmt.format(det_no, slice_no, v_max_signal, row_max_signal, signal, bgd, noise, snr)
                    if is_low_snr:
                        text += ' Low SNR'
                    print(text)
                if is_low_snr:
                    continue
                pt_u_coords, pt_v_coords, is_error = Opt02._get_trace_coordinates(slice_image, slice_row_min,
                                                                                  v_max_signal, is_alpha)
                u_mean = np.mean(pt_u_coords)       # Initialise trace parameters (overwritten by fit)
                v_fid = np.mean(pt_v_coords)
                popt, pcov = None, None
                if is_error and Globals.is_debug('high'):
                    text = fmt.format(det_no, slice_no, v_max_signal, row_max_signal, signal, bgd, noise, snr)
                    print(text + ' Gaussian fit failed')

                p0_guess = [0.]*4
                n_func_pars = len(p0_guess)

                sigma_filter = True
                while sigma_filter:
                    n_data_points = len(pt_u_coords)
                    if n_data_points < n_func_pars:
                        sigma_filter = False
                        # print('Skipping polynomial fit')
                        continue
                    u_mean = np.mean(pt_u_coords)
                    p0_guess[0] = u_mean
                    with warnings.catch_warnings(record=True):
                        popt, pcov = curve_fit(Globals.polynomial, pt_u_coords, pt_v_coords, p0=p0_guess)
                    vs = Globals.polynomial(pt_u_coords, *popt)
                    displacements = np.array(pt_v_coords) - vs
                    clip_limit = np.std(displacements) * sigma_clip
                    indices = np.argwhere(np.abs(displacements) > clip_limit)
                    if indices.shape[0] == 0:
                        sigma_filter = False
                        # print(slice_no, n_data_points, 'No points to delete')
                    else:
                        idx = indices[0, 0]     # Remove points one at a time
                        # print(slice_no, n_data_points, pt_v_coords[idx], displacements[idx])
                        pt_u_coords = np.delete(pt_u_coords, idx, axis=0)
                        pt_v_coords = np.delete(pt_v_coords, idx, axis=0)
                    v_fid = Globals.polynomial(u_mean, *popt)[0]

                if popt is None:
                    # print(slice_no, n_data_points, 'No polynomial trace fit found')
                    continue
                lms_config = Globals.lms_config_template.copy()
                lms_config['opticon'] = opticon
                lms_config['pri_ang'] = pri_ang
                lms_config['ech_ang'] = ech_ang
                version = 'ait'
                lms_config_id = Globals.make_lms_config_id(lms_config)
                det_trace = {'lms_config_id': lms_config_id, 'version': version,
                             'type': trace_type,
                             'det_no': det_no, 'slice_no': slice_no,
                             'snr': snr, 'signal': signal,
                             'popt': popt, 'pcov': pcov, 'pt_u_coords': pt_u_coords, 'pt_v_coords': pt_v_coords,
                             'u_mean': u_mean, 'v_fid': v_fid, 'efp_x': efp_x, 'efp_y': efp_y, 'efp_w': efp_w,
                             'ref_ech_ord': ref_ech_ord, 'opticon': opticon, 'pri_ang': pri_ang, 'ech_ang': ech_ang,
                             'fits_name': name
                             }
                det_traces.append(det_trace)
        det_traces = OptTools._find_thetas(det_traces)          # Add detector rotation angle measurements
        print("- {:d} {:s} traces found with snr > {:4.1f}".format(len(det_traces), trace_type, snr_cut))
        if Globals.is_debug('high'):
            Opt02._print_det_traces(det_traces, trace_type)
        return det_traces

    @staticmethod
    def _print_det_traces(det_traces, type):
        print('Trace type = ', type)
        fmt = "{:>10s},{:>10s},{:>12s},{:>12s},{:>20s}"
        u_tag = 'row' if type == 'iso-lambda' else 'col' + '_mean'
        v_tag = 'col' if type == 'iso-alpha' else 'row' + '_fiducial'
        print(fmt.format('Det. no.', 'Slice no.', u_tag, v_tag, 'theta_disp / deg'))
        fmt = "{:10d},{:10d},{:12.3f},{:12.3f},{:20.3f}"
        for det_trace in det_traces:
            det_no = det_trace['det_no']
            slice_no = det_trace['slice_no']
            u_mean = det_trace['u_mean']
            v_fid = det_trace['v_fid']
            rot_angle = det_trace['theta']
            print(fmt.format(det_no, slice_no, u_mean, v_fid, rot_angle))
        return

    @staticmethod
    def _get_trace_coordinates(slice_image, rs_min, v_max, is_alpha):
        """ Now sample the trace at multiple u locations, find the centroid v_cen at each location and fit a
        polynomial to all v_cen.
        :param  slice_image ~130 row x 2048 section of the image matching a specific slice.
        :param  rs_min ~ Absolute row number of bottom left corner of slice_image in main image.
        :param  v_max ~ Row offset of brightest row with slice image.
        :param  is_alpha ~ True if the trace is iso-alpha, False otherwise.
        """
        (u_off, u_axis, v_axis) = (0, 1, 0) if is_alpha else (rs_min, 0, 1)
        u_count = slice_image.shape[u_axis]
        v_count = slice_image.shape[v_axis]

        n_samples = 10 if is_alpha else 5
        u_interval = u_count // (n_samples + 1)
        u_start, u_end = u_interval, u_count - u_interval
        u_list = list(range(u_start, u_end, u_interval))
        u_hw, v_hw = 5, 10       # Sample half width in along and across trace dimensions to fit gaussian.

        v1, v2 = v_max - v_hw, v_max + v_hw  # Across trace rows/cols to use for trace analysis
        v1, v2 = max(v1, 0), min(v2, v_count)
        pt_u_coords, pt_v_coords = [], []
        for u in u_list:
            # Find the trace coordinates by fitting gaussians
            u1, u2 = u - u_hw, u + u_hw
            u1, u2 = max(u1, 0), min(u2, u_count)
            sample_image = slice_image[v1:v2, u1:u2] if is_alpha else slice_image[u1:u2, v1:v2]
            z_vals = np.mean(sample_image, axis=u_axis)
            v_vals = np.array(list(range(v1, v2)))
            idx_max = np.argmax(z_vals)
            z_max = z_vals[idx_max]
            v_sigma = 1.0
            p0_guess = [z_max, v1 + v_hw, v_sigma]
            is_error = False
            try:
                gopt, gcov = curve_fit(Globals.gauss, v_vals, z_vals, p0=p0_guess)
                v_off = rs_min if is_alpha else 0
                v_cen = gopt[1] + v_off     # Get the row/col coordinate in the image frame.
                pt_u_coords.append(float(u + u_off))
                pt_v_coords.append(v_cen)
            except:
                a = 1

        return pt_u_coords, pt_v_coords, is_error

    @staticmethod
    def _find_trace_intersections(alpha_traces, lambda_traces, affines):
        """ Find the ray coordinates at the input focal plane (alpha, beta, lambda) and detector focal plane
        (det_no/mos_idx, column, row)
        """
        # Select traces for each slice
        util = Util()
        dist_coord = {}
        for det_no in range(1, 5):
            for alpha_trace in alpha_traces:
                # Filter out traces on other detectors
                if np.array(alpha_trace['det_no']) != det_no:
                    continue
                slice_no_alpha = alpha_trace['slice_no']
                a_config_id = alpha_trace['lms_config_id']
                print(a_config_id)
                for lambda_trace in lambda_traces:
                    if np.array(lambda_trace['det_no']) != det_no:
                        continue
                    slice_no_lambda = lambda_trace['slice_no']
                    if slice_no_alpha != slice_no_lambda:
                        continue
                    l_config_id = lambda_trace['lms_config_id']
                    print(l_config_id)
                    print()
                    if a_config_id == l_config_id:
                        if a_config_id in dist_coord:           # Add points to existing LMS configuration.
                            points = dist_coord[a_config_id]
                        else:                                   # New LMS configuration
                            lms_config = Globals.lms_config_template.copy()
                            lms_config['opticon'] = alpha_trace['opticon']
                            lms_config['pri_ang'] = alpha_trace['pri_ang']
                            lms_config['ech_ang'] = alpha_trace['ech_ang']
                            ref_ech_ord = alpha_trace['ref_ech_ord']
                            points = {'lms_config': lms_config, 'ref_ech_ord': ref_ech_ord,
                                      'det_no': [], 'slice_no': [], 'row': [], 'col': [],
                                      'efp_x': [], 'efp_y': [], 'efp_w': []}
                            dist_coord[a_config_id] = points
                        a_popt = np.array(alpha_trace['popt'])
                        l_popt = np.array(lambda_trace['popt'])
                        efp_x, efp_y, efp_w = alpha_trace['efp_x'], alpha_trace['efp_y'], lambda_trace['efp_w']
                        # Use an iterative method to find the intersection (y = a(x), x = w(y))
                        x, y, dx_res, dy_res = 1024., 1024., .01, .01
                        is_converged, count = False, 0
                        while not is_converged:
                            yp, = Globals.polynomial(x, *a_popt)
                            dy = math.fabs(yp - y)
                            xp, = Globals.polynomial(yp, *l_popt)
                            dx = math.fabs(xp - x)
                            x, y = xp, yp
                            count += 1
                            is_converged = dy < dy_res and dx < dx_res
                        points['det_no'].append(det_no)
                        points['slice_no'].append(slice_no_alpha)
                        points['col'].append(x)
                        points['row'].append(y)
                        points['efp_x'].append(efp_x)
                        points['efp_y'].append(efp_y)
                        points['efp_w'].append(efp_w)
                        dfp_xy = {'dfp_x': np.array(points['col']), 'dfp_y': np.array(points['row']),
                                  'det_nos': np.array(points['det_no'])}
                        mfp_xy = util.dfp_to_mfp(affines, dfp_xy)
                        points['mfp_x'], points['mfp_y'] = mfp_xy['mfp_x'], mfp_xy['mfp_y']
        return dist_coord

    @staticmethod
    def _print_dist_coord(filer, dist_coord, to_csv=False):
        path = None
        if to_csv:
            filename = 'LMS_ait_dist_coord.csv'
            path = filer.ray_trace_folder + '/' + filename
            file = open(path, 'w')
            orig_stdout = sys.stdout
            sys.stdout = file

        for config in dist_coord:
            coord = dist_coord[config]
            lms_config = coord['lms_config']
            opticon = lms_config['opticon']
            pri_ang = lms_config['pri_ang']
            ech_ang = lms_config['ech_ang']
            ref_ech_ord = coord['ref_ech_ord']

            if opticon == Globals.nominal:
                print("{:s},{:d}".format('Spectral order:', ref_ech_ord))
            print("{:s},{:10.3f}".format('Echelle angle:', ech_ang))
            print("{:s},{:10.3f}".format('Prism angle:', pri_ang))
            print()
            fmt = "{:12s},{:12s},{:12s},{:12s},{:12s},{:12s},"
            print(fmt.format('slice_no', 'efp_w', 'efp_x', 'efp_y', 'mfp_x', 'mfp_y'))
            fmt = "{:12d},{:12.3f},{:12.3f},{:12.3f},{:12.3f},{:12.3f},"
            for i in range(0, len(coord['det_no'])):
                slice_no = coord['slice_no'][i]
                efp_w = coord['efp_w'][i]
                efp_x = coord['efp_x'][i]
                efp_y = coord['efp_y'][i]
                mfp_x = coord['mfp_x'][i]
                mfp_y = coord['mfp_y'][i]
                print(fmt.format(slice_no, efp_w, efp_x, efp_y, mfp_x, mfp_y))
        if to_csv:
            file.close()
            sys.stdout = orig_stdout
        return path