#!/usr/bin/env python
"""
Decorators for use in all LMS projects.  Currently just includes @debug

@author: Alistair Glasse

Update:
"""
import math

import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning

from lms_globals import Globals
from lms_mosaic import Mosaic
from lmsaiv_opt_tools import OptTools
from lmsaiv_plot import Plot
from lms_filer import Filer


class Opt02:


    def __init__(self):
        return

    @staticmethod
    def dist(title, as_built, **kwargs):
        test_name = 'lms_opt_02'
        opticon = Globals.nominal
        inc_tags = ['lms_opt_02', opticon[0:3]]         # Tokens to identify image files.

        # Dictionary of detector rotation angles and intra-detector gaps, encoded in the affine mfp to dfp transforms.
        # The elements will be created as they are calculated, with names (d=dispersion, n/m=det_no) theta_dn, gap_nm
        det_position = {}

        print('Analysing test data for {:s}, {:s}'.format(test_name, title))
        print()
        slice_map = as_built["slice_map_{:s}".format(opticon)]
        if Globals.is_debug('medium'):
            Plot.mosaic(slice_map, title='Slice Map', cmap='hsv', mask=(0.0, 'black'))

        # Set up a deliberately misaligned detector
        test_det_no = 2
        test_det_offset = 0, 0
        test_det_rot_deg = 0.3
        test_img_rot_deg = -test_det_rot_deg
        print('Deliberately misaligning detector')
        print('det_no =       {:d}'.format(test_det_no))
        print('det_offset =   {:d}, {:d} arcsec'.format(test_det_offset[0], test_det_offset[1]))
        print('det_rotation = {:5.3f} deg.'.format(test_det_rot_deg))
        print()

        print('1. Derive detector rotation and offset differences from trace data. ')
        skip_iso_alpha = False
        if not skip_iso_alpha:
            # ---------- ISO-ALPHA -------------------
            # Start with data loading and background subtraction.  The iso-alphas use the WCU BB and the lm_pinhole
            # mask.  The background under the trace is then the emission spectrum of the mask (at 300 K).  We remove
            # this by moving the pinhole out of the LMS field using the CFO chopper.
            alpha_traces = []

            # Load background images (BB flat field)
            print('Background')
            bgd_mosaics = Filer.read_mosaic_list(inc_tags + ['flat'])
            bgd_mosaic = OptTools.transform_detector_image(bgd_mosaics[0],
                                                           det_no=test_det_no,
                                                           xy_pix=test_det_offset,
                                                           angle=test_img_rot_deg)
            if Globals.is_debug('medium'):
                Plot.mosaic(bgd_mosaic, title='WCU lm_pinhole mask background', cmap='hot')

            sig_mosaics = Filer.read_mosaic_list(inc_tags + ['iso_alpha'])
            for sig_mosaic in sig_mosaics:
                print()
                print("Processing mosaic file {:s}".format(sig_mosaic[0]))
                sig_mosaic = OptTools.transform_detector_image(sig_mosaic,
                                                               det_no=test_det_no,
                                                               xy_pix=test_det_offset,
                                                               angle=test_img_rot_deg)
                # Extract iso-alpha traces for all slices (snr_cut=0.)
                tra_mosaic = Mosaic.diff_mosaics(sig_mosaic, bgd_mosaic)
                psf_alpha_traces = Opt02._extract_det_traces(tra_mosaic, 'iso-alpha', slice_map,
                                                             snr_cut=0.)
                # ..then fit a gaussian to all solid detections to find the single trace for the PSF centroid.
                # The slice plus beta phase should ideally(!) match the WCU derived efp_y parameter.
                alpha_trace_list = Opt02._find_beta_phase(psf_alpha_traces, snr_cut=1.)
                if Globals.is_debug('low'):
                    print()
                    print('PSF iso-alpha traces found')
                    fmt = "{:>12s},{:>12s},{:>12s},"
                    print(fmt.format('Det. no.', 'Slice no.', 'Beta phase'))
                    fmt = "{:12d},{:12d},{:12.3f},"
                    for alpha_trace in alpha_trace_list:
                        det_no = alpha_trace['det_no']
                        slice_no = alpha_trace['slice_no']
                        beta_phase = alpha_trace['beta_phase']
                        print(fmt.format(det_no, slice_no, beta_phase))

                if Globals.is_debug('high'):
                    Plot.mosaic(tra_mosaic, title='Signal')
                    Plot.mosaic(sig_mosaic, title='Signal - Bgd', cmap='hot')
                if Globals.is_debug('low'):
                    overlay = {'type': 'det_traces', 'data': alpha_trace_list}
                    Plot.mosaic(sig_mosaic, title='Background subtracted', cmap='hot', overlay=overlay)
                alpha_traces = alpha_traces + alpha_trace_list         # 1 alpha traces per detector

            if Globals.is_debug('medium'):
                # Find pairs of iso-alphas for short and long wave traces.
                slice_pairs = []
                n_traces = len(alpha_traces)
                for idx1 in range(n_traces):
                    trace_idx1 = alpha_traces['trace_idx'][idx1]
                    slice_1 = alpha_traces['slice_no'][idx1]
                    for idx2 in range(idx1 + 1, n_traces):
                        slice_2 = alpha_traces['slice_no'][idx2]
                        if slice_1 == slice_2:
                            trace_idx2 = alpha_traces['trace_idx'][idx2]
                            slice_pairs.append((slice_1, trace_idx1, trace_idx2))
                            continue

                # Define a fiducial position at the mosaic centre in polynomial pixel column coordinates
                col_half_gap = 1000. * Globals.det_gap / Globals.nom_pix_pitch
                fmt = None
                if Globals.is_debug('low'):
                    fmt = "{:>10s},{:>10s},{:>12s},{:>20s},{:>24s},{:>24s},{:>24s},"
                    print(fmt.format('Det_SW', 'Slice', 'Row_SW', 'Row_LW - Row_SW at', 'Equivalent rotation', 'SW dispersion wrt', 'LW dispersion wrt'))
                    print(fmt.format('      ', 'No.', 'fiducial', 'mosaic centre', '(cw det 23) / deg.', 'row / deg.', 'row / deg.'))
                    fmt = "{:>10d},{:>10d},{:>12.2f},{:>20.2f},{:>24.3f},{:>24.3f},{:>24.3f},"
                for slice_no, trace_idx1, trace_idx2 in slice_pairs:
                    det_1 = alpha_traces['det_no'][trace_idx1]
                    idx_sw, idx_lw = trace_idx1, trace_idx2
                    det_sw = det_1
                    if det_1 in [2, 4]:
                        idx_sw, idx_lw = trace_idx2, trace_idx1
                        det_sw = alpha_traces['det_no'][idx_sw]
                    col_fid_sw = Globals.det_format[0] + col_half_gap
                    col_fid_lw = -col_half_gap
                    popt_sw = alpha_traces['popt'][idx_sw]
                    popt_lw = alpha_traces['popt'][idx_lw]
                    row_fid_sw = Globals.polynomial(col_fid_sw, *popt_sw)
                    row_fid_lw = Globals.polynomial(col_fid_lw, *popt_lw)
                    delta_row_fid = row_fid_lw - row_fid_sw
                    # Calculate the angle between polynomials at the mosaic centre.
                    deg_rad = 180./math.pi
                    rel_rot_angle = -deg_rad * delta_row_fid / Globals.det_format[0]
                    # Calculate the angle between the alpha trace (dispersion) and the detector row at the centre column
                    col_cen = Globals.det_format[0] / 2
                    disp_row_angle_sw = deg_rad * Globals.polynomial(col_cen, *popt_sw, gradient=True)
                    disp_row_angle_lw = deg_rad * Globals.polynomial(col_cen, *popt_lw, gradient=True)

                    print(fmt.format(det_sw, slice_no, row_fid_sw, delta_row_fid, rel_rot_angle, disp_row_angle_sw, disp_row_angle_lw))

        # -----------------------
        # ISO-LAMBDA
        print('2. Extract iso-lambda traces to measure the intra-detector gap and the line spread function.')
        print('   The gap calculation will assume that the laser lines are spaced according to a smooth polynomial.')

        mosaics = Filer.read_mosaic_list(inc_tags + ['iso_lambda'])
        print(0)
        lambda_traces = []

        for mosaic in mosaics:
            print()
            print("Processing mosaic file {:s}".format(mosaic[0]))
            mosaic = OptTools.transform_detector_image(mosaic,
                                                       det_no=test_det_no,
                                                       xy_pix=test_det_offset,
                                                       angle=test_img_rot_deg)
            lt_wave = mosaic[1]['HIERARCH ACHG LASER WAVE']
            if Globals.is_debug('medium'):
                Plot.mosaic(mosaic, title='laser_wavelength = ' + "{:10.3f}".format(lt_wave))
            # Find the gap between detectors as a function of row number.
            snr_cut = 20
            mos_lambda_traces = Opt02._extract_det_traces(mosaic, 'iso-lambda', slice_map, snr_cut, lt_wave=lt_wave)
            lambda_traces = lambda_traces + mos_lambda_traces

        # Find the intra-detector gap from a list of column position v wavelength for tunable laser spectral lines.
        Opt02._find_detector_gaps(lambda_traces)

        dist_coord = Opt02._find_trace_intersections(alpha_traces, lambda_traces)
        Opt02._print_dist_coord(dist_coord)
        as_built['dist_coord'] = dist_coord
        return as_built

    @staticmethod
    def _find_beta_phase(all_alpha_traces, snr_cut=1.):
        """ Find the beta phase, the offset in slices of a PSF from the slice centre, from a list of alpha tracs for
        a sliced PSF image.
        :param all_alpha_traces:
        :return:
        """
        alpha_trace_list = []
        for det_no in range(1, 5):
            signal_list, slice_no_list, det_alpha_traces = [], [], []
            is_detected = False
            for alpha_trace in all_alpha_traces:
                if det_no != alpha_trace['det_no']:
                    continue
                signal_list.append(alpha_trace['signal'])
                slice_no_list.append(alpha_trace['slice_no'])
                det_alpha_traces.append(alpha_trace)
                snr = alpha_trace['snr']
                is_detected = True if snr > snr_cut else is_detected

            if len(signal_list) == 0 or not is_detected:
                continue
            signals, slice_nos = np.array(signal_list), np.array(slice_no_list)
            sig_max = np.amax(np.array(signals))
            idx_max = np.argmax(signals)
            slice_no_peak = slice_nos[idx_max]
            alpha_trace = det_alpha_traces[idx_max].copy()
            psf_alpha_sigma = 2.5
            p0_guess = [sig_max, slice_no_peak, psf_alpha_sigma]
            try:
                gopt, gcov = curve_fit(Globals.gauss, slice_nos, signals, p0=p0_guess)
            except:
                print('!! Error finding beta-phase !!')
            beta_phase = gopt[1] - slice_no_peak + 0.5       # Phase = 0.5 for source centred in slice
            alpha_trace['beta_phase'] = beta_phase
            alpha_trace_list.append(alpha_trace)
        return alpha_trace_list

    @staticmethod
    def _find_detector_gaps(lambda_traces):
        gap_data = {}
        for lambda_trace in lambda_traces:
            slice_no = lambda_trace['slice_no']
            if slice_no not in gap_data:
                gap_data[slice_no] = {'lams_l': [], 'lams_r': [], 'cols_l': [], 'cols_r': []}
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
            lam_popt_l, _ = curve_fit(Globals.polynomial, cols_l, lams_l, p0=[0., 0.])
            lam_mid_l = Globals.polynomial(2048, *lam_popt_l)       # Wavelength of RH edge of left hand detector
            col_popt_r, _ = curve_fit(Globals.polynomial, lams_r, cols_r, p0=[0., 0.])
            col_gap = -1. * Globals.polynomial(lam_mid_l, *col_popt_r)
            gap_sample['col_gap'] = col_gap

        # Find angle between 1,2 and 3,4 by linear fitting to the intra-detector gap as a function of row number.
        for det_nos in ['12', '34']:
            col_gap_list, row_list = [], []
            for slice_no in gap_data:
                gap_sample = gap_data[slice_no]
                if str(gap_sample['det_no']) in det_nos:
                    col_gap_list.append(gap_sample['col_gap'])
                    row_list.append(gap_sample['u_mean'])

            rows = np.array(row_list)
            gaps = np.array(col_gap_list)
            mean_gap = np.mean(gaps)

            linear_guess = [mean_gap, 0.]
            try:
                popt, pcov = curve_fit(Globals.polynomial, rows, gaps, p0=linear_guess)
            except OptimizeWarning:
                print('Warning: polynomial fit failed')
            gradient = Globals.polynomial(mean_gap, *popt, gradient=True)
            gradient_err = np.sqrt(pcov[1][1])
            deg_rad = 180. / math.pi
            theta = math.atan(gradient) * deg_rad
            theta_err = gradient_err * deg_rad / (1 + gradient ** 2)
            print(det_nos, theta, theta_err)
        return

    @staticmethod
    def _extract_det_traces(mosaic, trace_type, slice_map, snr_cut,
                            lt_wave=None):
        """ Extract iso-alpha or iso-lambda traces from a spectral image.
        :param mosaic: Data tuple (name, _, image list)  Background subtracted trace mosaic
        :param trace_type: 'iso-alpha' or 'iso-lambda'
        :param slice_map:
        :param kwargs:
        :return:
        """
        popt, pcov = None, None
        mos_name, mos_primary_header, mos_hdus = mosaic

        is_alpha = trace_type == 'iso-alpha'

        efp_w = -1. if is_alpha else lt_wave
        efp_x = mosaic[1]['HIERARCH ACHG WCU X']
        efp_y = mosaic[1]['HIERARCH ACHG WCU Y']
        ref_ech_ord = mosaic[1]['HIERARCH ACHG REF_ECH_ORD']
        opticon = mosaic[1]['HIERARCH AIT OPTICON']
        pri_ang = mosaic[1]['HIERARCH AIT PRI_ANG']
        ech_ang = mosaic[1]['HIERARCH AIT ECH_ANG']
        config_id = "pa{:04d}_ea{:04d}_reo{:02d}".format(int(1000 * pri_ang), int(1000 * ech_ang), ref_ech_ord)

        det_traces = []
        _, _, slice_map_hdus = slice_map
        fmt = ''
        if Globals.is_debug('low'):
            rowcol_tag = 'Row' if is_alpha else 'Column'
            sli_tag = rowcol_tag + ' in slice'
            abs_tag = rowcol_tag + ' in image'
            print()
            print("Slice by slice {:s} trace extraction".format(trace_type))
            fmt = "{:>12s},{:>12s},{:>20s},{:>20s},{:>12s},{:>12s},{:>12s},{:>12s}"
            print(fmt.format('Detector', 'Slice ', 'Brightest', 'Brightest', 'Signal', 'Bgd', 'Noise', 'SNR'))
            print(fmt.format('Number  ', 'Number',  sli_tag   , abs_tag    ,   'DN/sec', 'DN/sec', '1 sigma', '-'))
            fmt = "{:>12d},{:>12d},{:>20d},{:>20d},{:>12.2f},{:>12.2f},{:>12.2f},{:>12.1f}"

        for hdu in mos_hdus:
            det_no = int(hdu.header['ID'].strip())
            mos_idx = Globals.mos_idx[det_no]
            # Read in image and replace low (dark) values with the median/background level
            raw_image = np.array(hdu.data)
            bgd = np.median(raw_image)
            image = np.where(raw_image < 0.2 * bgd, bgd, raw_image)

            slice_map_data = np.array(slice_map_hdus[mos_idx].data)
            snu = np.unique(slice_map_data)
            slice_nos = [int(s) for s in snu if s > 0.]
            # Start by extracting the data for each slice.
            for slice_no in slice_nos:
                ism = int(0.5 * Globals.intra_slice_gap)        # Intra-slice margin
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
                if Globals.is_debug('low'):
                    text = fmt.format(det_no, slice_no, v_max_signal, row_max_signal, signal, bgd, noise, snr)
                    if is_low_snr:
                        text += ' Low SNR'
                    print(text)
                if is_low_snr:
                    continue
                pt_u_coords, pt_v_coords, is_error = Opt02._get_trace_coordinates(slice_image, slice_row_min,
                                                                                  v_max_signal, is_alpha)
                if is_error:
                    text = fmt.format(det_no, slice_no, v_max_signal, row_max_signal, signal, bgd, noise, snr)
                    print(text + ' Gaussian fit failed')

                u_mean = np.mean(pt_u_coords)
                p0_guess = [u_mean, 0., 0., 0.]
                n_func_pars = len(p0_guess)
                n_data_points = len(pt_u_coords)
                if n_data_points < n_func_pars:
                    continue
                try:
                    popt, pcov = curve_fit(Globals.polynomial, pt_u_coords, pt_v_coords, p0=p0_guess)
                except OptimizeWarning as e:           # OptimizeWarning: RuntimeError
                    print('!! Error finding polynomial trace fit !!' + e)
                v_fid = Globals.polynomial(u_mean, *popt)
                det_trace = {'config_id': config_id, 'type': trace_type,
                             'det_no': det_no, 'slice_no': slice_no,
                             'snr': snr, 'signal': signal,
                             'popt': popt, 'pcov': pcov, 'pt_u_coords': pt_u_coords, 'pt_v_coords': pt_v_coords,
                             'u_mean': u_mean, 'v_fid': v_fid, 'efp_x': efp_x, 'efp_y': efp_y, 'efp_w': efp_w,
                             'ref_ech_ord': ref_ech_ord, 'opticon': opticon, 'pri_ang': pri_ang, 'ech_ang': ech_ang,
                             'fits_name': mos_name
                             }
                det_traces.append(det_trace)
        det_traces = OptTools._find_thetas(det_traces)          # Add detector rotation angle measurements
        print("- {:d} {:s} traces found with snr > {:4.1f}".format(len(det_traces), trace_type, snr_cut))
        if Globals.is_debug('low'):
            Opt02._print_det_traces(det_traces, 'iso-lambda')
        return det_traces

    @staticmethod
    def _print_det_traces(det_traces, type):
        print('Trace type = ', type)
        fmt = "{:>10s},{:>10s},{:>12s},{:>12s},{:>20s}"
        u_tag = 'row' if type == 'iso-lambda' else 'col' + '_mean'
        v_tag = 'col' if type == 'iso-alpha' else 'row' + '_fiducial'
        print(fmt.format('Det. no.', 'Slice no.', u_tag, v_tag, 'theta / deg'))
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
                is_error = True
        return pt_u_coords, pt_v_coords, is_error

    @staticmethod
    def _find_trace_intersections(alpha_traces, lambda_traces):
        """ Find the ray coordinates at the input focal plane (alpha, beta, lambda) and detector focal plane
        (det_no/mos_idx, column, row)
        """
        # Select traces for each slice
        dist_coord = {}
        for det_no in range(1, 5):
            for alpha_trace in alpha_traces:
                # Filter out traces on other detectors
                if np.array(alpha_trace['det_no']) != det_no:
                    continue
                slice_no_alpha = alpha_trace['slice_no']
                a_config_id = alpha_trace['config_id']
                for lambda_trace in lambda_traces:
                    if np.array(lambda_trace['det_no']) != det_no:
                        continue
                    slice_no_lambda = lambda_trace['slice_no']
                    if slice_no_alpha != slice_no_lambda:
                        continue
                    l_config_id = lambda_trace['config_id']
                    if a_config_id == l_config_id:
                        if a_config_id in dist_coord:          # Add points to existing LMS configuration.
                            points = dist_coord[a_config_id]
                        else:                               # New LMS configuration
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
                        yp = Globals.polynomial(x, *a_popt)
                        dy = math.fabs(yp - y)
                        xp = Globals.polynomial(yp, *l_popt)
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
        return dist_coord

    @staticmethod
    def _print_dist_coord(dist_coord):

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
            print(fmt.format('slice', 'efp_w', 'efp_x', 'efp_y', 'det_x', 'det_y'))
            fmt = "{:12d},{:12.3f},{:12.3f},{:12.3f},{:12.3f},{:12.3f},"
            for i in range(0, len(coord['det_no'])):
                slice_no = coord['slice_no'][i]
                efp_w = coord['efp_w'][i]
                efp_x = coord['efp_x'][i]
                efp_y = coord['efp_y'][i]
                mfp_x = coord['col'][i]
                mfp_y = coord['row'][i]
                print(fmt.format(slice_no, efp_w, efp_x, efp_y, mfp_x, mfp_y))
