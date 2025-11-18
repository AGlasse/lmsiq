#!/usr/bin/env python
"""
@author: Alistair Glasse

Update:
"""
import math
import numpy as np
from astropy import units as u
from scipy.optimize import least_squares
from scipy.optimize import curve_fit
from lms_globals import Globals


class Util:

    n_lines_config = -1

    def __init__(self):
        return

    @staticmethod
    def print_list(title, val_list):
        text = title + ' - '
        for val in val_list:
            text += "{:d}, ".format(val)
        print(text)
        return

    @staticmethod
    def write_text_file(transform_file, text_block):
        tf_file = open(transform_file, 'w')
        tf_file.write(text_block)
        tf_file.close()
        return tf_file

    @staticmethod
    def read_text_file(text_file, **kwargs):
        n_hdr_lines = kwargs.get('n_hdr_lines', 0)
        all_lines = open(text_file, 'r').read().splitlines()
        lines = all_lines[n_hdr_lines:]
        return lines

    @staticmethod
    def waves_to_phases(waves, ech_ords):
        d = Globals.rule_spacing * Globals.ge_refractive_index
        phases = waves * ech_ords / (2.0 * d)
        return phases      # phases

    @staticmethod
    def phases_to_waves(phases, ech_ords):
        d = Globals.rule_spacing * Globals.ge_refractive_index
        waves = phases * 2.0 * d / ech_ords
        return waves

    @staticmethod
    def apply_affine_transform(x, y, aff):
        """
        @author Alistair Glasse
        """
        n_pts = len(x)
        unity = np.full(n_pts, 1.)

        xy = np.array([x, y, unity])
        uv = aff @ xy
        return uv[0], uv[1]

    @staticmethod
    def apply_svd_distortion(x, y, a, b):
        """
        @author Alistair Glasse
        21/2/17   Create to encapsulate distortion transforms
        Apply a polynomial transform pair (A,B or AI,BI) to an array of points
        affine = True  Apply an affine version of the transforms (remove all non-linear terms)
        """
        dim = a.shape[0]
        n_pts = len(x)

        exponent = np.array([range(0, dim), ] * n_pts).transpose()

        xmat = np.array([x, ] * dim)
        xin = np.power(xmat, exponent)
        ymat = np.array([y, ] * dim)
        yin = np.power(ymat, exponent)

        xout = np.zeros(n_pts)
        yout = np.zeros(n_pts)
        for i in range(0, n_pts):
            xout[i] = yin[:, i] @ a @ xin[:, i]
            yout[i] = yin[:, i] @ b @ xin[:, i]
        return xout, yout

    @staticmethod
    def read_raytrace_file(raytrace_file, **kwargs):
        debug = kwargs.get('debug', False)

        fmt = "Reading raytrace and transform data from file= {:s}"
        print(fmt.format(raytrace_file))
        lines = open(raytrace_file, 'r').read().splitlines()
        line_iter = iter(lines)
        configs, transforms, ray_coords = [], [], []
        eof = False
        while not eof:
            try:
                line = next(line_iter)
            except StopIteration:
                break
            tokens = line.split(',')
            ea, pa = float(tokens[1]), float(tokens[3])
            line = next(line_iter)
            tokens = line.split(',')
            slice_no, spifu_no = int(tokens[1]), int(tokens[3])
            config = [ea, pa, slice_no, spifu_no]
            configs.append(config)

            line = next(line_iter)
            tokens = line.split(',')
            n_mats, n_terms, n_layers = int(tokens[1]), int(tokens[3]), int(tokens[5])
            matrices = []
            for i in range(0, n_mats):
                next(line_iter)
                matrix = np.zeros((n_terms, n_terms, n_layers))
                for row in range(0, n_terms):
                    line = next(line_iter)
                    tokens = line.split(',')
                    idx = 0
                    for layer in range(0, n_layers):
                        for col in range(0, n_terms):
                            val = float(tokens[idx])
                            idx += 1
                            matrix[row, col, layer] = val
                matrices.append(matrix)
            transforms.append(matrices)
            line = next(line_iter)
            tokens = line.split(',')
            n_pts = int(tokens[1])
            line = next(line_iter)
            tokens = line.split(',')
            series, column = {}, {}

            for j, token in enumerate(tokens[:-1]):
                name = token.strip(' ')
                series[name] = []
                column[j] = name
            for i in range(0, n_pts):
                line = next(line_iter)
                tokens = line.split(',')
                for j, token in enumerate(tokens[:-1]):
                    val = float(tokens[j])
                    col = column[j]
                    series[col].append(val)
            for name in series:
                val_list = series[name]
                series[name] = np.array(val_list)
            ray_coords.append(series)
        return np.array(configs), np.array(transforms), ray_coords

    @staticmethod
    def extract_configs(tf_list, **kwargs):
        """ Extract unique configuration data (echelle angle, wavelength etc.)
        from a transform list.
        :param tf_list:
        :return:
        """
        config_list = []
        for i in range(0, 10):
            config_list.append([])
        wave_list = []
        for i in range(0, 4):
            wave_list.append([])
        stats_list = []
        for i in range(0, 4):
            stats_list.append([])

        tf_iter = iter(tf_list)
        tf_row = next(tf_iter)
        eof = False
        while not eof:
            tokens = tf_row.split(',')
            for i in range(0, 6):
                val = float(tokens[i])
                config_list[i-0].append(val)
            for i in range(12, 16):
                val = float(tokens[i])
                wave_list[i-12].append(val)
            for i in range(16, 20):
                val = float(tokens[i])
                stats_list[i-16].append(val)

            config_id_old = tokens[5]
            end_of_config = False
            i_mat = 0
            while not end_of_config:
                matrix_id_old = tokens[6]
                matrix = []
                end_of_matrix = False
                while not end_of_matrix:
                    row = []
                    for i in range(8, 12):
                        val = float(tokens[i])
                        row.append(val)
                    matrix.append(row)
                    try:
                        tf_row = next(tf_iter)
                    except StopIteration:
                        eof = True
                        break
                    tokens = tf_row.split(',')
                    matrix_id = tokens[6]
                    end_of_matrix = matrix_id != matrix_id_old
                config_list[6 + i_mat].append(matrix)
                i_mat += 1
                if eof:
                    break
                config_id = tokens[5]
                end_of_config = config_id != config_id_old
        # Convert lists to numpy arrays
        configs = []
        for config in config_list:
            configs.append(np.array(config))
        wave_limits = []
        for waves in wave_list:
            wave_limits.append(np.array(waves))
        stats = []
        for stat in stats_list:
            stats.append(np.array(stat))
        return configs, wave_limits, stats

    @staticmethod
    def find_polyfit(x, coeffs, **kwargs):
        debug = kwargs.get('debug', False)
        order = kwargs.get('order', 1)
        _, ncr, _ = coeffs.shape
        poly = np.zeros((ncr, ncr, order+1))
        for i in range(0, ncr):
            for j in range(0, ncr):
                y = coeffs[:, i, j]
                if debug:
                    print(x, y)
                poly[i, j] = np.polyfit(x, y, order)
        return poly

    @staticmethod
    def find_dispersion(traces, configuration):
        dw = 0.001  # Choose a wavelength interval of 1 nanometre (approx. 4000 / 50 = 80 pixels)

        slice_no, wave, prism, ech_angle, order, im_pix_size = configuration
        for trace in traces:
            is_order = order in trace.unique_ech_ords
            if not is_order:
                continue
            is_slice = slice_no in trace.unique_slices
            if not is_slice:
                continue
            for tf in trace.slice:
                tf_config, tf_matrices, _, _, _ = tf
                tf_ech_ord, tf_slice_no, tf_spifu_no = tf_config
                if order != tf_ech_ord and slice_no != tf_slice_no:
                    continue
                a, b, ai, bi = tf_matrices
                w1 = wave
                w2 = wave + dw
                phases = trace.waves_to_phase(np.array([w1, w2]), np.array([order, order]))
                alpha_middle = 0.0  # FP2 coordinate
                alphas = [alpha_middle, alpha_middle]
                det_x, _ = trace.apply_svd_distortion(np.array(phases), np.array(alphas), a, b)
                dx = det_x[1] - det_x[0]
                dw_dx = dw / dx  # micron per mm
                dw_lmspix = dw_dx * Globals.nom_pix_pitch / 1000.0  # micron per (nominal) pixel
        return dw_lmspix

    @staticmethod
    def print_stats(st_file, data):
        fmt1 = "{:15s}{:15s}"
        fmt2 = "{:20s}{:20s}{:20s}"
        fmt3 = "{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}"
        # Write header
        hdr = fmt1.format("Transform", "Polynomial")
        hdr = hdr + fmt2.format("Mean offset", "Std deviation", "Maximum")
        print(hdr)
        st_file.write(hdr + '\n')
        hdr = fmt1.format("matrix terms", "fit terms")
        hdr = hdr + fmt3.format("X", "Y", "X", "Y", "X", "Y")
        print(hdr)
        st_file.write(hdr + '\n')
        n_terms, poly_order, offset_xy = data
        micron_mm = 1000.0
        for i in range(0, len(offset_xy)):
            x = micron_mm * offset_xy[i, :, 0, :]
            y = micron_mm * offset_xy[i, :, 1, :]
            mx, my = np.mean(x), np.mean(y)
            vx, vy = np.std(x), np.std(y)
            zx, zy = np.max(x), np.max(y)

            fmt4 = "{:10d}{:15d}"
            text = fmt4.format(n_terms, poly_order+1)
            fmt5 = "{:15.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}"
            text += fmt5.format(mx, my, vx, vy, zx, zy)
            print(text)
            st_file.write(text + '\n')
        st_file.close()
        return

    @staticmethod
    def filter_transform_list(transform_list, **kwargs):
        """ Filter transform data to select a configuration value (e.g. keyword slice=13).
        """
        filtered_transform_list = []
        for transform in transform_list:
            cfg = transform.lms_configuration | transform.slice_configuration
            is_match = True
            for key in kwargs:
                is_match = False if cfg[key] != kwargs[key] else is_match
            if is_match:
                filtered_transform_list.append(transform)
        return filtered_transform_list

    @staticmethod
    def parse_slice_locations(slice_locs, **kwargs):
        boresight = kwargs.get('boresight', False)      # True = Extract boresight slice only
        tgt_slice = int(slice_locs[0])
        slice_idents = []
        if boresight:
            slice_folder = "slice_{:d}/".format(tgt_slice)
            slice_label = ", slice {:d}".format(tgt_slice)
            slice_idents.append((tgt_slice, slice_folder, slice_label))
            return tgt_slice, 1, slice_idents

        n_slices = int(slice_locs[1])
        slice_lo = tgt_slice - int(n_slices / 2)
        for slice_idx in range(n_slices):
            slice_no = slice_lo + slice_idx
            slice_folder = "slice_{:d}/".format(slice_no)
            slice_label = ", slice {:d}".format(slice_no)
            slice_idents.append((slice_no, slice_folder, slice_label))
        if slice_locs[2] == 'ifu':
            slice_folder, slice_label = "ifu/", ", IFU image"
            slice_idents.append((-1, slice_folder, slice_label))
        return tgt_slice, n_slices, slice_idents

    @staticmethod
    def read_polyfits_file(poly_file):
        pf = open(poly_file, 'r')
        lines = pf.read().splitlines()
        pf.close()
        poly = []
        for line in lines[1:]:
            tokens = line.split(',')
            if len(tokens) < 2:
                break
            poly_row = []
            for token in tokens[:-1]:
                val = float(token.strip(' '))
                poly_row.append(val)
            poly.append(poly_row)
        return np.array(poly)

    @staticmethod
    def _decode_tf_list_orders(tf_list):
        ech_ords = []
        for tf in tf_list:
            tokens = tf.split(',')
            eo = int(tokens[1])
            if eo not in ech_ords:
                ech_ords.append(eo)
        return ech_ords

    @staticmethod
    def get_transform_text(transform):
        text = ''
        n_mat = len(transform)
        is_first_matrix = True
        for mat_name in transform:
            matrix = np.atleast_3d(transform[mat_name])
            if is_first_matrix:
                n_rows, n_cols, n_layers = matrix.shape
                fmt2b = "n_mats=,{:4d},n_terms=,{:4d},n_layers=,{:4d},\n"
                text += fmt2b.format(n_mat, n_rows, n_layers)
                is_first_matrix = False
            fmt = "Matrix {:s},\n"
            text += fmt.format(mat_name)
            for row in range(0, n_rows):
                for layer in range(0, n_layers):
                    for col in range(0, n_cols):
                        val = matrix[row, col, layer]
                        text += "{:15.6e},".format(val)
                    text += "   "
                text += '\n'
        return text

    @staticmethod
    def get_term_values(transforms, slice_no, spifu_no, debug=False):
        """ Find the transforms which map the centre of slice 13 (alpha = 0.0) onto the EFP.
        """
        phase = 0.                      # Across slice fractional displacement
        y = Util.slice_to_efp_y(slice_no, phase)
        efp_y = np.array([y.value])     # Location in EFP.
        efp_x = np.array([0.])
        efp_bs = {'efp_y': efp_y, 'efp_x': efp_x}
        if debug:
            print("Slice no= {:d}, Sp. IFU no= {:d}".format(spifu_no))
            fmt = "{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}"
            print(fmt.format('pri_ang', 'ech_ang', 'efp_x', 'efp_y', 'mfp_x', 'mfp_y', 'w'))
        term_values = {'slice_no': slice_no, 'spifu_no': spifu_no,
                       'pri_ang': [], 'ech_ang': [],
                       'mfp_y': [], 'w_bs': [], 'ech_ords': [], 'matrices': []}
        for transform in transforms:
            lms_config = transform.lms_configuration
            slice_cfg = transform.slice_configuration
            if (slice_cfg['slice_no'] != slice_no) or (slice_cfg['spifu_no'] != spifu_no):
                continue
            # Calculate the wavelength at fps_x = 0.0 (the mosaic column direction mid-line)
            w_min, w_max = slice_cfg['w_min'], slice_cfg['w_max']
            mfp_w_list, mfp_x_list = [], []
            for efp_w in np.linspace(w_min, w_max, 100):
                efp_bs['efp_w'] = np.array([efp_w])
                mfp_bs, _ = Util.efp_to_mfp(transform, efp_bs)
                mfp_x = mfp_bs['mfp_x'][0]
                mfp_x_list.append(mfp_x)
                mfp_w_list.append(efp_w)

            mfp_w = np.interp(0., np.array(mfp_x_list), np.array(mfp_w_list))
            mfp_y = mfp_bs['mfp_y'][0]
            if debug:
                fmt = "{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f},{:10.3f}"
                print(fmt.format(lms_config['pri_ang'], lms_config['ech_ang'],
                                 efp_bs['efp_x'][0], efp_bs['efp_y'][0],
                                 mfp_x, mfp_y, mfp_w))
            term_values['pri_ang'].append(lms_config['pri_ang'])
            term_values['ech_ang'].append(lms_config['ech_ang'])
            term_values['ech_ords'].append(slice_cfg['ech_ord'])
            term_values['mfp_y'].append(mfp_y)
            term_values['w_bs'].append(mfp_w)
            term_values['matrices'].append(transform.matrices)
        return term_values

    @staticmethod
    def find_closest_transforms(wave, opticon, svd_transforms):
        """ Find the transforms (one per slice) which position a specific wavelength closest to the centre of the
        detector mosaic.  For the extended mode, this is defined as order 24, slice 13

        Step 1. Find the ~boresight transform for slice 13 (For spifu selected, use spifu = 1 and ech_ord = 24)
        """
        bs_transform = None
        w_off_min = 1000.*u.nm
        for svd_transform in svd_transforms:
            config = svd_transform.configuration
            slice_no = config['slice_no']
            if slice_no != 13:
                continue
            spifu_no = config['spifu_no']
            if spifu_no > 1:        # Catches both nominal and spifu configurations
                continue
            w_min, w_max = config['w_min']*u.micron, config['w_max']*u.micron
            w_off = abs(0.5 * (w_max + w_min) - wave)
            if w_off <= w_off_min:
                bs_transform = svd_transform
                w_off_min = w_off

        # Step 2, select all transforms with same configuration as the boresight. For spifu, use optimum order
        bs_cfg = bs_transform.configuration
        opt_cfg_id = bs_cfg['cfg_id']
        opt_transforms = {}
        ech_ords = []             # List of unique echelle orders in optimum transforms
        for svd_transform in svd_transforms:
            cfg = svd_transform.configuration
            if cfg['cfg_id'] != opt_cfg_id:
                continue
            ech_ord = cfg['ech_ord']
            # Option to only use specific orders for specific spifu_slices
            spec_order = False
            if spec_order:
                if opticon == Globals.extended:
                    spifu_no = cfg['spifu_no']
                    spifu_slice = {23: (1, 4), 24: (2, 5), 25: (3, 6)}
                    valid_spifu_nos = spifu_slice[ech_ord]
                    if spifu_no not in valid_spifu_nos:
                        continue
            if ech_ord not in ech_ords:
                ech_ords.append(ech_ord)
            slice_no = cfg['slice_no']
            spifu_no = cfg['spifu_no']
            slice_id = Globals.slice_id_fmt.format(opticon, opt_cfg_id, ech_ord, slice_no, spifu_no)
            # print('Adding slice_id= ', slice_id)
            opt_transforms[slice_id] = svd_transform
        return opt_transforms, ech_ords

    @staticmethod
    def out_of_bounds(fp_id, x, y):       # Check if out of bounds
        bounds = Globals.fp_bounds[fp_id]
        x1, x2, y1, y2 = bounds
        in_x = np.any(np.logical_and(np.less(x1, x), np.greater(x2, x)))
        in_y = np.any(np.logical_and(np.less(y1, y), np.greater(y2, y)))
        in_bounds = in_x and in_y
        return not in_bounds

    @staticmethod
    def efp_y_to_slice(efp_y):
        """ Convert EFP y coordinate (mm) into a slice number and phase (the offset from the slice centre as
        a fraction of the slice width.
        """
        efp_slice_width = Globals.beta_slice.to(u.arcsec) / Globals.efp_as_mm
        n_slices = Globals.n_lms_slices
        y_s = efp_y / efp_slice_width
        slice_coord = n_slices // 2 + y_s
        slice_no = n_slices // 2 + y_s.astype(int)
        phase = slice_coord - slice_no
        return slice_no, phase

    @staticmethod
    def slice_to_efp_y(slice_no, phase):
        """ Return the EFP y coordinate of the slice centre (in mm).
        """
        efp_slice_width = Globals.beta_slice.to(u.arcsec) / Globals.efp_as_mm
        n_slices = Globals.n_lms_slices
        slice = slice_no + phase
        efp_y = (slice - n_slices // 2) * efp_slice_width
        return efp_y

    @staticmethod
    def efp_to_dfp(transform, affines, efp_points):
        mfp_points, _ = Util.efp_to_mfp(transform, efp_points)
        dfp_points = Util.mfp_to_dfp(affines, mfp_points)
        return dfp_points

    @staticmethod
    def mfp_to_dfp(affines, mfp_points):
        """ Note, mfp increases towards decreasing wavelength.  Det 1/3 are then at a shorter wavelength (larger
        mfp) than dets 2/4.  Also, slice 1 is at the minimum of mfp_y.
        """
        x = mfp_points['mfp_x']
        y = mfp_points['mfp_y']
        idx24 = np.argwhere(x < 0.)
        idx13 = np.argwhere(x > 0.)
        idx34 = np.argwhere(y < 0.)
        idx12 = np.argwhere(y > 0.)
        idx1 = np.intersect1d(idx12, idx13)
        idx2 = np.intersect1d(idx12, idx24)
        idx3 = np.intersect1d(idx13, idx34)
        idx4 = np.intersect1d(idx24, idx34)

        idx_list = [idx1, idx2, idx3, idx4]
        det_nos, dfp_x, dfp_y = [], [], []
        for i, idx in enumerate(idx_list):
            n_idx = len(idx)
            if n_idx == 0:
                continue
            d_nos = [i+1]*n_idx
            affine_fwd = affines[i]
            u, v = Util.apply_affine_transform(x[idx], y[idx], affine_fwd)
            det_nos += d_nos
            dfp_x += list(u)
            dfp_y += list(v)
        dfp_points = {'det_nos': np.array(det_nos), 'dfp_x': np.array(dfp_x), 'dfp_y': np.array(dfp_y)}
        return dfp_points

    @staticmethod
    def dfp_to_mfp(affines, dfp_points):
        """ Transform points from detector to the mosaic focal plane.  Note that all points are assumed
        to come from the same detector (as specified in the first point).
        """
        det_nos = dfp_points['det_nos']
        dfp_x = dfp_points['dfp_x']
        dfp_y = dfp_points['dfp_y']
        mfp_x, mfp_y = [], []
        n_pts, = det_nos.shape
        for det_idx in range(4):
            det_no = det_idx + 1
            det_no_array = np.full(n_pts, det_no)
            idx, = np.nonzero(np.equal(det_nos, det_no_array))
            n_idx, = idx.shape
            if n_idx == 0:
                continue
            affine_rev = affines[det_idx + 4]
            u, v = Util.apply_affine_transform(dfp_x[idx], dfp_y[idx], affine_rev)
            mfp_x += list(u)
            mfp_y += list(v)

        mfp_points = {'mfp_x': np.array(mfp_x), 'mfp_y': np.array(mfp_y)}
        return mfp_points

    @staticmethod
    def efp_to_mfp(transform, efp_points):
        """ Transform a point in the LMS entrance focal plane onto the detector mosaic.
        Note that the transform will normally have been verified as non-vignetting
        for this point in the EFP.
        """
        config, matrices = transform.slice_configuration, transform.matrices
        ech_ord = config['ech_ord']

        alphas, efp_y, efp_w = efp_points['efp_x'], efp_points['efp_y'], efp_points['efp_w']
        ech_ords = np.full(len(efp_w), ech_ord)
        phases = Util.waves_to_phases(efp_w, ech_ords)

        a, b = matrices['a'], matrices['b']
        mfp_x, mfp_y = Util.apply_svd_distortion(phases, alphas, a, b)
        oob = Util.out_of_bounds('mfp', mfp_x, mfp_y)       # Check if out of bounds
        mfp_points = {'mfp_x': mfp_x, 'mfp_y': mfp_y}
        return mfp_points, oob

    @staticmethod
    def dfp_to_efp(transform, affines, dfp_points):
        mfp_points = Util.dfp_to_mfp(affines, dfp_points)
        efp_points = Util.mfp_to_efp(transform, mfp_points)
        return efp_points

    @staticmethod
    def mfp_to_efp(transform, mfp_points, slice_phase=0.):
        """ Transform a point on the LMS detector (det_x, det_y) (w, efp_x, efp_y) onto the
        detector mosaic (det_x, det_y).  Note that the transform will normally have been verified as non-vignetting
        for this point in the EFP.
        slice_phase: Intra-slice position is degenerate with wavelength. It can be passed explicitly here, with
                     the default being to assume the centre of the slice in the EFP (slice_phase = 0.).
        """
        ech_ord = transform.slice_configuration['ech_ord']
        slice_no = transform.slice_configuration['slice_no']
        mfp_x, mfp_y = mfp_points['mfp_x'], mfp_points['mfp_y']

        matrices = transform.matrices
        ai, bi = matrices['ai'], matrices['bi']
        phases, alphas = Util.apply_svd_distortion(mfp_x, mfp_y, ai, bi)

        ech_ords = np.full(alphas.shape, ech_ord)
        waves = Util.phases_to_waves(phases, ech_ords)

        efp_y_val = Util.slice_to_efp_y(slice_no, slice_phase)
        efp_y = np.full(alphas.shape, efp_y_val)

        efp_points = {'efp_x': alphas, 'efp_y': efp_y, 'efp_w': waves}
        return efp_points

    @staticmethod
    def sort(ref, unsort):
        """ Re-order arrays in list 'unsort' using the index list from sorting array 'ref' in ascending order.
        """
        indices = np.argsort(ref)
        sort = []
        for array in unsort:
            sort.append(array[indices])
        return ref[indices], tuple(sort)

    @staticmethod
    def lookup_transform_fit(config, prism_angle, transform_fits):
        ech_ord, slice_no, spifu_no = config
        n_slices, n_mats, n_terms, _, n_polys = transform_fits.shape
        i_slice = int(slice_no - 1)
        matrix_fits = []
        for i_mat in range(0, n_mats):
            matrix_fit = np.zeros((n_terms, n_terms))
            for row in range(0, n_terms):
                for col in range(0, n_terms):
                    poly = transform_fits[i_slice, i_mat, row, col, :]
                    fit = np.poly1d(poly)
                    term = fit(prism_angle)
                    matrix_fit[row, col] = term
            matrix_fits.append(matrix_fit)
        return tuple(matrix_fits)

    @staticmethod
    def print_poly_transform(transform, **kwargs):
        label = kwargs.get('label', '---')
        print(label)
        for matrix, mat_name in zip(transform, Globals.matrix_names):
            n_terms = matrix.shape[0]
            print("Matrix {:s}".format(mat_name))
            for r in range(0, n_terms):
                line = ""
                for c in range(0, n_terms - r):
                    token = "{:10.3e}".format(matrix[r, c])
                    line = line + token
                print(line)
        return

    @staticmethod
    def find_fwhm_lin(x, y):
        """ Find the FWMH of a profile by using linear interpolation to find where it crosses the half-power
        level.  Referred to in summary files as the 'linear' FWHM etc.
        """
        ymax = np.amax(y)
        yh = ymax / 2.0
        yz = np.subtract(y, yh)       # Search for zero crossing
        iz = np.where(yz > 0)
        il = iz[0][0]
        ir = iz[0][-1]
        xl = x[il-1] - yz[il-1] * (x[il] - x[il-1]) / (yz[il] - yz[il-1])
        xr = x[ir] - yz[ir] * (x[ir+1] - x[ir]) / (yz[ir+1] - yz[ir])
        return xl, xr, yh

    @staticmethod
    def solve_svd_distortion(x_in, y_in, x_out, y_out, order, inverse=False):
        # @author Alistair GlasseTea Temim
        # Converted to python from Ronayette's original IDL code by Temim, 21/2/17
        #   1. Changed input parameters to allow use with any data set (Glasse)
        # make a polynomial fit of the imaged grid point
        # see note on MIRI calibration
        # if keyword set inverse: swaps Xout and Xin, Yout and Yin
        # to compute the coefficients for the transform from FPA to entrance plane
        # INPUTS: XYin           Input xo,Y coordinates
        #          XYout
        #          order         Polynomial order for transform
        #          inverse       True = Calculate inverse transform (out -> in)
        #
        # OUTPUTS: amat, bmat, the distortion coefficients
        xi = x_in.copy()
        yi = y_in.copy()
        xo = x_out.copy()
        yo = y_out.copy()
        if inverse:
            temp = xo
            xo = xi
            xi = temp.copy()
            temp = yo
            yo = yi.copy()
            yi = temp.copy()

        dim = order + 1

        ind = np.arange(0, dim*dim)
        rows = np.floor(ind / dim)
        cols = ind % dim
        mask_ld = rows + cols <= order      # Mask lower diagonal to zero (False)
        mask = np.array([mask_ld.astype(int),] * x_out.size).transpose()
        exponent = np.array([range(0, dim), ] * x_out.size).transpose()

        mx = np.tile(np.power(xi, exponent), (dim, 1))
        my = np.repeat(np.power(yi, exponent), dim, axis=0)
        mxy = np.multiply(mx, my)
        m = np.multiply(mxy, mask)

        v, w, u = np.linalg.svd(m, full_matrices=False)
        n_w = w.size
        wp = np.zeros((n_w, n_w))
        svd_cutoff = Globals.svd_cutoff            # Only use singular values >= svd_cutoff
        for k in range(0, n_w):
            if abs(w[k]) >= svd_cutoff:
                wp[k, k] = 1.0 / w[k]
            else:
                if w[k] > 1.0e-16:
                    print("!! Clipping singular value = {:5.3e} !!".format(w[k]))
        a = xo @ u.T @ wp @ v.T     # (Was wp.T, but wp is square diagonal)
        amat = np.reshape(a, (dim, dim))
        b = yo @ u.T @ wp @ v.T
        bmat = np.reshape(b, (dim, dim))
        return amat, bmat

    @staticmethod
    def test_out_and_back(filer, opticon, date_stamp):
        print('lms_distort - Evaluating transforms')

        # Define EFP coordinates and target wavelength for spectrum
        efp_x_cen, efp_y_cen, efp_w_cen = 0., 0., 4.65
        tgt_slice_nos, test_phases = Util.efp_y_to_slice([efp_y_cen])
        tgt_slice_no = tgt_slice_nos[0]

        # Generate test spectrum for the wavelength/order which is closest to the mfp_y = 0. column.
        affines = filer.read_fits_affine_transform(date_stamp)
        svd_transforms = filer.read_svd_transforms()
        opt_transforms = Util.find_closest_transforms(efp_w_cen, opticon, svd_transforms)
        opt_transform = opt_transforms[tgt_slice_no]

        # Check out and back for the wavelength min, max and centre of the spectrum.
        w_min = opt_transform['configuration']['w_min']
        w_max = opt_transform['configuration']['w_max']

        n_pts = 30  # No. of wavelength samples in EFP
        # Create EFP data cube for extended/background emission.
        efp_ws = np.linspace(w_min, w_max, n_pts)
        efp_ys = np.zeros(n_pts)  # EFP across slice coordinate
        efp_fs = np.zeros(n_pts)  # Flux values
        efp_as_mm = Globals.efp_as_mm
        alpha_fov = Globals.alpha_fov
        efp_xmax = alpha_fov / efp_as_mm
        xh = efp_xmax / 2.
        efp_xs = np.linspace(-xh, +xh, n_pts)
        efp_out = {'efp_x': efp_xs, 'efp_y': efp_ys, 'efp_w': efp_ws, 'efp_f': efp_fs}

        test_transforms = []

        config = opt_transform['configuration']
        slice_no = config['slice']
        matrices = opt_transform['matrices']
        mfp_points = Util.efp_to_mfp(opt_transform, efp_out)
        dfp_points = Util.mfp_to_dfp(affines, mfp_points)
        mfp_back = Util.dfp_to_mfp(affines, dfp_points)
        efp_back = Util.mfp_to_efp(opt_transform, mfp_back)

        mfp_x, mfp_y = mfp_points['mfp_x'], mfp_points['mfp_y']
        det_nos, dfp_x, dfp_y = dfp_points['det_nos'], dfp_points['dfp_x'], dfp_points['dfp_y']
        pri_ang = config['pri_ang']
        ech_ang = config['ech_ang']
        ech_ord = config['ech_ord']

        fmt = "{:>7s},{:>7s},{:>7s},{:>8s},{:>9s},{:>9s},{:>12s},{:>9s},{:>12s},{:>9s},{:>9s},{:>8s},{:>8s},{:>15s},{:>12s}"
        print(fmt.format('out', 'out', 'out', 'prism', 'echelle', 'echelle',
                         'mosaic', 'mosaic', 'det', 'det', 'det', 'back', 'back', 'back-out', 'back-out'))
        print(fmt.format('efp_x', 'efp_y', 'efp_w', 'angle', 'angle', 'order',
                         'mfp_x', 'mfp_y', 'no.', 'dfp_x', 'dfp_y', 'efp_x', 'efp_y', 'delta_efp_x', 'delta_efp_y'))
        print(fmt.format('mm', 'mm', 'micron', 'deg.', 'deg.', '-',
                         'mm', 'mm', '-', 'pix', 'pix', 'mm', 'mm', 'mm', 'mm'))
        fmt1 = "{:7.3f},{:7.3f},{:7.3f},{:8.3f},{:9.3f},{:9d},"
        fmt2 = "{:12.3f},{:9.3f},{:12d},{:9.1f},{:9.1f},{:8.3f},{:8.3f},{:15.3f},{:12.3f}"
        fmt = fmt1 + fmt2
        for i in range(0, n_pts):
            efp_out_x, efp_out_y = efp_out['efp_x'][i], efp_out['efp_y'][i]
            efp_back_x, efp_back_y = efp_back['efp_x'][i], efp_back['efp_y'][i]
            delta_efp_x = efp_back_x - efp_out_x
            delta_efp_y = efp_back_y - efp_out_y
            print(fmt.format(efp_out_x, efp_out_y, efp_out['efp_w'][i],
                             pri_ang, ech_ang, ech_ord,
                             mfp_x[i], mfp_y[i],
                             det_nos[i], dfp_x[i], dfp_y[i],
                             efp_back_x, efp_back_y,
                             delta_efp_x, delta_efp_y))
        return
