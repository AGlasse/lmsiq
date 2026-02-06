#!/usr/bin/env python
"""
Decorators for use in all LMS projects.  Currently just includes @debug

@author: Alistair Glasse

Update:
"""
import math

import numpy as np
from scipy.optimize import curve_fit

from lms_globals import Globals
from lmsaiv_opt_tools import OptTools
from lmsaiv_plot import Plot
from lms_filer import Filer


class Opt02:


    def __init__(self):
        return

    @staticmethod
    def dist(title, as_built, **kwargs):
        test_name = 'lms_opt_02'
        print('Analysing test data for {:s}, {:s}'.format(test_name, title))
        print()
        slice_map = as_built['slice_map']
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
        # ---------- ISO-ALPHA -------------------
        # Start with data loading and background subtraction.
        alpha_traces = None
        inc_tags = ['lms_opt_02', 'iso_alpha']
        exc_tags = ['bgd']
        bgd_mosaics = Filer.read_mosaic_list(inc_tags + exc_tags)
        print()

        alpha_traces_list = []
        bgd_mosaic = OptTools.transform_detector_image(bgd_mosaics[0],
                                                       det_no=test_det_no,
                                                       xy_pix=test_det_offset,
                                                       angle=test_img_rot_deg)
        if Globals.is_debug('low'):
            Plot.mosaic(bgd_mosaic, title='background', cmap='hot')

        sig_mosaics = Filer.read_mosaic_list(inc_tags, exc_tags)
        for sig_mosaic in sig_mosaics:
            print("Processing mosaic file {:s}".format(sig_mosaic[0]))
            sig_mosaic = OptTools.transform_detector_image(sig_mosaic,
                                                           det_no=test_det_no,
                                                           xy_pix=test_det_offset,
                                                           angle=test_img_rot_deg)
            mosaic = OptTools.subtract_mosaics(sig_mosaic, bgd_mosaic)

            # Extract iso-alpha traces as a list of polynomials.
            alpha_traces = OptTools.extract_det_traces(mosaic, 'alpha', slice_map)
            alpha_traces_list.append(alpha_traces)
            if Globals.is_debug('low'):
                Plot.mosaic(sig_mosaic, title='signal', cmap='hot')
                Plot.mosaic(mosaic, title='bgd subtracted', cmap='hot', overlay=alpha_traces)

        if Globals.is_debug('high'):
            # Find pairs of iso-alphas for short and long wave traces.
            slice_pairs = []
            n_traces = len(alpha_traces['trace_idx'])
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

        mosaics = Filer.read_mosaic_list(['lms_opt_02', 'iso_lambda', 'sig_bs'])
        for mosaic in mosaics:
            mosaic = OptTools.transform_detector_image(mosaic,
                                                       det_no=test_det_no,
                                                       xy_pix=test_det_offset,
                                                       angle=test_img_rot_deg)
        bgd_mosaics = Filer.read_mosaic_list(['lms_opt_02', 'iso_lambda', 'bgd'])
        bgd_mosaic = OptTools.transform_detector_image(bgd_mosaics[0],
                                                       det_no=test_det_no,
                                                       xy_pix=test_det_offset,
                                                       angle=test_img_rot_deg)
        if Globals.is_debug('low'):
            Plot.mosaic(bgd_mosaic, title=bgd_mosaic[0] + r'\n background', cmap='hot')
        lambda_traces_list = []
        lambda_traces = None
        for mosaic in mosaics:
            lambda_traces = OptTools.extract_det_traces(mosaic, 'lambda', slice_map)
            lambda_traces_list.append(lambda_traces)
            if Globals.is_debug('low'):
                n_traces = len(lambda_traces['trace_idx'])
                print("   - found {:d} iso-lambda traces".format(n_traces))
                Plot.mosaic(mosaic, title='signal', cmap='hot')
                Plot.mosaic(mosaic, title='bgd subtracted', cmap='hot', overlay=lambda_traces)

        # Estimate width of gap between dets 1/2 and 3/4
        gap_samples = []
        slice_nos = lambda_traces['slice_no']
        n_traces = len(slice_nos)
        u_slice_nos = np.unique(slice_nos)
        for slice_no in u_slice_nos:
            idx1, = np.where(slice_nos == slice_no)
            mos_idxs = np.array(lambda_traces['mos_idx'])[idx1]
            # Get the coordinates of the gaussian peak for the first point in each slice.
            pt_u_list, pt_v_list = [], []

            for i in range(0, n_traces):
                pt_u_list.append(lambda_traces['pt_u_coords'][i][0])
                pt_v_list.append(lambda_traces['pt_v_coords'][i][0])
            pt_us, pt_vs = np.array(pt_u_list)[idx1], np.array(pt_v_list)[idx1]
            u_means = np.array(lambda_traces['u_mean'])[idx1]
            u_mean = np.mean(u_means)
            uni_mos_idxs = np.unique(mos_idxs)

            lw_mos_idx = uni_mos_idxs[1]
            idx2 = np.where(lw_mos_idx == mos_idxs)
            pt_vs[idx2] += Globals.det_format[0]
            y = np.sort(pt_vs)
            col_intervals = y[1:] - y[:-1]
            gap = np.median(col_intervals) - np.amin(col_intervals)
            gap_sample = uni_mos_idxs[0], uni_mos_idxs[1], slice_no, u_mean, gap
            gap_samples.append(gap_sample)

        # Calculate angle between detectors from gap data.
        gap_data = np.array(gap_samples)
        sw_mos_idxs = np.int64(gap_data[:, 0])
        uni_sw_mos_idxs = np.unique(sw_mos_idxs)
        gap_thetas = {}
        for sw_mos_idx in uni_sw_mos_idxs:
            lw_mos_idx = sw_mos_idx + 1
            idx, = np.where(sw_mos_idxs == sw_mos_idx)
            rows = gap_data[idx, 3]
            gaps = gap_data[idx, 4]
            mean_gap = np.mean(gaps)
            linear_guess = [mean_gap, 0.]
            popt, pcov = curve_fit(Globals.polynomial, rows, gaps, p0=linear_guess)
            gradient = Globals.polynomial(mean_gap, *popt, gradient=True)
            gradient_err = np.sqrt(pcov[1][1])
            deg_rad = 180. / math.pi
            theta = math.atan(gradient) * deg_rad
            theta_err = gradient_err * deg_rad / (1 + gradient ** 2)
            gap_thetas[sw_mos_idx] = sw_mos_idx, lw_mos_idx, theta, theta_err, mean_gap

        # Print theta_01 and theta_23
        if Globals.is_debug('low'):
            fmt = "{:>12s},{:>12s},{:>12s},{:>12s},{:>12s},"
            print(fmt.format('SW Det.', 'LW Det.', 'theta(SW-> LW)', 'theta error', 'Mean gap'))
            print(fmt.format('Mos. index', 'Mos. index', 'deg.', 'deg.', 'pix.'))
            fmt = "{:>12.0f},{:>12.0f},{:>12.3f},{:>12.3f},{:>12.2f},"
            for key in gap_thetas.keys():
                theta = gap_thetas[key]
                sw_mos_idx, lw_mos_idx, theta, theta_err, mean_gap = theta
                print(fmt.format(sw_mos_idx, lw_mos_idx, theta, theta_err, mean_gap))
            Plot.gap_samples(gap_samples, gap_thetas)

        if Globals.is_debug('low'):
            fmt = "{:>12s},{:>12s},{:>12s},{:>12s},{:>12s},"
            print(fmt.format('SW Det.', 'LW Det.', 'Slice', 'Row', 'Gap'))
            print(fmt.format('Mos. index', 'Mos. index', 'Number', '/pixel', '/pixel'))
            fmt = "{:>16.0f},{:>16.0f},{:>12d},{:>12.2f},{:>12.2f},"
            for gap_sample in gap_samples:
                umi0, umi1, slice_no, u_mean, gap = gap_sample
                print(fmt.format(umi0, umi1, slice_no, u_mean, gap))

        dist_points = Opt02.find_trace_intersections(alpha_traces, lambda_traces)

        return as_built

    @staticmethod
    def find_trace_intersections(alpha_traces, lambda_traces):
        """ Find the ray coordinates at the input focal plane (alpha, beta, lambda) and detector focal plane
        (det_no/mos_idx, column, row)
        """
        # Select traces for each slice
        points = {'det_no': [], 'slice_no': [], 'row': [], 'col': []}

        fmt= ''
        if Globals.is_debug('low'):
            print('iso-alpha, iso-lambda intersections found for file {:s}'.format(alpha_traces['name']))
            fmt = "{:>8s},{:>8s},{:>12s},{:>12s},"
            print(fmt.format('Det', 'Slice', 'Column', 'Row'))
            print(fmt.format('No.', 'No.', 'pix.', 'pix.'))
            fmt = "{:8d},{:8d},{:12.3f},{:12.3f},"

        for det_no in range(1, 5):
            cond = np.array(alpha_traces['det_no']) == det_no
            ab = np.asarray(cond)
            a_det_filter = np.array(alpha_traces['det_no']) == det_no
            w_det_filter = np.array(lambda_traces['det_no']) == det_no
            for slice_no in range(1, 29):
                a_sli_filter = np.array(alpha_traces['slice_no']) == slice_no
                w_sli_filter = np.array(lambda_traces['slice_no']) == slice_no
                a_filter = np.logical_and(a_det_filter, a_sli_filter)
                w_filter = np.logical_and(w_det_filter, w_sli_filter)
                a_idxs = np.array([i for i, boolean in enumerate(a_filter) if boolean])
                w_idxs = np.array([i for i, boolean in enumerate(w_filter) if boolean])
                if a_idxs.size == 0 or w_idxs.size == 0:
                    continue
                a_popts = np.array(alpha_traces['popt'])[a_idxs]
                w_popts = np.array(lambda_traces['popt'])[w_idxs]
                for a_popt in a_popts:
                    for w_popt in w_popts:
                        # We choose to use an iterative method to find the intersection (y = a(x), x = w(y))
                        x, y, dx_res, dy_res = 1024., 1024., .01, .01
                        is_converged, count = False, 0
                        while not is_converged:
                            yp = Globals.polynomial(x, *a_popt)
                            dy = math.fabs(yp - y)
                            xp = Globals.polynomial(yp, *w_popt)
                            dx = math.fabs(xp - x)
                            x, y = xp, yp
                            count += 1
                            is_converged = dy < dy_res and dx < dx_res
                        points['det_no'].append(det_no)
                        points['slice_no'].append(slice_no)
                        points['col'].append(x)
                        points['row'].append(y)
                        if Globals.is_debug('low'):
                            print(fmt.format(det_no, slice_no, x, y))

        dist_points = None
        return dist_points

    @staticmethod
    def psf(test_name, title, as_built, **kwargs):
        print('PSF = ')
        return