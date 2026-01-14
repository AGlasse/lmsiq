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
    def run(test_name, title, as_built, **kwargs):
        print('Analysing test data for {:s}, {:s}'.format(test_name, title))

        slice_map = as_built['slice_map']
        if Globals.is_debug('medium'):
            Plot.mosaic(slice_map, title='Slice Map', cmap='hsv', mask=(0.0, 'black'))

        # Set up a misaligned detector
        test_det_no = 2
        test_det_offset = 0, 0
        test_det_rot_deg = 0.3

        print('1. Extract iso-lambda traces to measure the intra-detector gap and the line spread function.')
        print('   The gap calculation will assume that the laser lines are spaced according to a smooth polynomial.')
        print('   - !! Analysis of iso_lambdas to find detector gap not yet implemented !!')

        iso_lambdas = Filer.read_mosaic_list([test_name, 'iso_lambda'])
        lambda_traces = None
        for iso_lambda in iso_lambdas:
            iso_lambda = OptTools.transform_detector_image(iso_lambda,
                                                           det_no=test_det_no,
                                                           xy_pix=test_det_offset,
                                                           angle=test_det_rot_deg)
            if Globals.is_debug('low'):
                Plot.mosaic(iso_lambda, title=iso_lambda[0], cmap='hot')
            lambda_traces = OptTools.extract_det_traces(iso_lambda, 'lambda', slice_map)
            if Globals.is_debug('low'):
                n_traces = len(lambda_traces['trace_idx'])
                print("   - found {:d} iso-lambda traces".format(n_traces))
                Plot.mosaic(iso_lambda, title=iso_lambda[0], cmap='hot', overlay=lambda_traces)
        # Estimate width of gap between dets 1/2 and 3/4
        gap_samples = []
        slice_nos = lambda_traces['slice_no']
        uni_mod_idxs = None
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
            uni_mod_idxs = np.unique(mos_idxs)

            lw_mos_idx = uni_mod_idxs[1]
            idx2 = np.where(lw_mos_idx == mos_idxs)
            pt_vs[idx2] += Globals.det_format[0]
            y = np.sort(pt_vs)
            col_intervals = y[1:] - y[:-1]
            gap = np.median(col_intervals) - np.amin(col_intervals)
            gap_sample = uni_mod_idxs[0], uni_mod_idxs[1], slice_no, u_mean, gap
            gap_samples.append(gap_sample)

        # Calculate angle between detectors from gap data.
        gap_data = np.array(gap_samples)
        sw_mos_idxs = gap_data[:, 0]
        uni_sw_mos_idxs = np.unique(sw_mos_idxs)
        thetas = {}
        for sw_mos_idx in uni_sw_mos_idxs:
            idx, = np.where(sw_mos_idxs == sw_mos_idx)
            rows = gap_data[idx, 3]
            gaps = gap_data[idx, 4]
            mean_gap = np.mean(gaps)
            linear_guess = [mean_gap, 0.]
            popt, pcov = curve_fit(Globals.polynomial, rows, gaps, p0=linear_guess)
            gradient = Globals.polynomial(mean_gap, *popt, gradient=True)
            theta = math.atan(gradient) * 180. / math.pi
            thetas[sw_mos_idx] = theta

        # Print gaps
        if Globals.is_debug('low'):
            Plot.gap_samples(gap_samples, thetas)

        if Globals.is_debug('low'):
            fmt = "{:>16s},{:>12s},{:>12s},{:>12s},"
            print(fmt.format('SW Detector', 'Slice', 'Row', 'Gap'))
            print(fmt.format('Index', 'Number', '/pixel', '/pixel'))
            fmt = "{:>16d},{:>12d},{:>12.2f},{:>12.2f},"
            for gap_sample in gap_samples:
                uni_mod_idxs[0], slice_no, u_mean, gap = gap_sample
                print(fmt.format(uni_mod_idxs[0], slice_no, u_mean, gap))

        print('2. Derive detector rotation and offset differences from iso-alpha data. ')
        do_plot = True
        # Extract iso-alpha traces as a list of polynomials
        alpha_traces = None
        inc_tags = [test_name, 'iso_alpha']
        iso_alphas = Filer.read_mosaic_list(inc_tags)
        for iso_alpha in iso_alphas[1:]:
            print("\nMosaic file = {:s}".format(iso_alpha[0]))
            iso_alpha = OptTools.transform_detector_image(iso_alpha,
                                                          det_no=test_det_no,
                                                          xy_pix=test_det_offset,
                                                          angle=test_det_rot_deg)
            alpha_traces = OptTools.extract_det_traces(iso_alpha, 'alpha', slice_map)
            if do_plot:
                Plot.mosaic(iso_alpha, title=iso_alpha[0], cmap='hot')
                Plot.mosaic(iso_alpha, title=iso_alpha[0], cmap='hot', overlay=alpha_traces)
                do_plot = False
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

            if Globals.is_debug('low'):
                print(fmt.format(det_sw, slice_no, row_fid_sw, delta_row_fid, rel_rot_angle, disp_row_angle_sw, disp_row_angle_lw))
        inc_tags = [test_name, 'cfo_pnh']
        iso_alpha_mosaics = Filer.read_mosaic_list(inc_tags)
        return as_built
