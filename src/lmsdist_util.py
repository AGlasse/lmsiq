#!/usr/bin/env python
"""
@author: Alistair Glasse

Update:
"""
import math
import numpy as np
from lms_globals import Globals
from scipy.optimize import least_squares


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
    def waves_to_phases(waves, ech_orders):
        d = Globals.rule_spacing * Globals.ge_refractive_index
        sin_aoi = waves * ech_orders / (2.0 * d)
        phases = np.arcsin(sin_aoi)
        return phases

    @staticmethod
    def phases_to_waves(phases, ech_orders):
        d = Globals.rule_spacing * Globals.ge_refractive_index
        sin_aoi = np.sin(phases)
        waves = sin_aoi * 2.0 * d / ech_orders
        return waves

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
            is_order = order in trace.unique_ech_orders
            if not is_order:
                continue
            is_slice = slice_no in trace.unique_slices
            if not is_slice:
                continue
            for tf in trace.slice_objects:
                tf_config, tf_matrices, _, _, _ = tf
                tf_ech_order, tf_slice_no, tf_spifu_no = tf_config
                if order != tf_ech_order and slice_no != tf_slice_no:
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
    def filter_transform_list(all_configs, **kwargs):
        """ Filter transform data with matching echelle order and slice number.
        """
        ech_order = kwargs.get('ech_order', -1.)
#        ech_angle = kwargs.get('ech_angle', -1.)
        slice_no = kwargs.get('slice_no', -1.)
        spifu_no = kwargs.get('spifu_no', -1.)
        idx_eo = all_configs[0] == ech_order
        idx_sno = all_configs[3] == slice_no
        idx_spifu = all_configs[4] == spifu_no
        idx = np.logical_and(idx_eo, idx_sno, idx_spifu)
        configs = []
        for i in range(0, len(all_configs)):
            series = np.compress(idx, all_configs[i], axis=0)
            configs.append(series)
        return configs

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
        ech_orders = []
        for tf in tf_list:
            tokens = tf.split(',')
            eo = int(tokens[1])
            if eo not in ech_orders:
                ech_orders.append(eo)
        return ech_orders

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
    def efp_y_to_slice(efp_y):
        """ Convert EFP y coordinate (mm) into a slice number and phase (the offset from the slice centre as
        a fraction of the slice width
        """
        efp_slice_width = .001 * Globals.beta_mas_pix / Globals.efp_as_mm
        n_slices = Globals.n_lms_slices
        slice_coord = n_slices // 2 + efp_y / efp_slice_width
        slice_no = n_slices // 2 + int(efp_y / efp_slice_width)
        phase = slice_coord - slice_no
        return slice_no, phase

    @staticmethod
    def slice_to_efp_y(slice_no, phase):
        efp_slice_width = .001 * Globals.beta_mas_pix / Globals.efp_as_mm
        n_slices = Globals.n_lms_slices
        slice = slice_no + phase
        efp_y = (slice - n_slices // 2) * efp_slice_width
        return efp_y

    @staticmethod
    def efp_to_dfp(transform, efp_points):
        """ Transform a point in the LMS entrance focal plane onto the detector mosaic.
        Note that the transform will normally have been verified as non-vignetting
        for this point in the EFP.
        """
        config = transform['configuration']
        ech_ord = config['ech_ord']

        alphas, efp_y, efp_w = efp_points['efp_x'], efp_points['efp_y'], efp_points['efp_w']
        ech_orders = np.full(efp_w.shape, ech_ord)
        phases = Util.waves_to_phases(efp_w, ech_orders)

        xp1, xp3, xp5 = config['xp1'], config['xp3'], config['xp5']
        yp1, yp3, yp5 = config['yp1'], config['yp3'], config['yp5']
        xp = [xp1, xp3, xp5]
        yp = [yp1, yp3, yp5]

        matrices = transform['matrices']
        a, b = matrices['a'], matrices['b']
        det_x_svd, det_y_svd = Util.apply_svd_distortion(phases, alphas, a, b)
        res_x = Util.offset_error_function(xp, det_x_svd, 0.)
        res_y = Util.offset_error_function(yp, det_y_svd, 0.)
        det_x, det_y = det_x_svd + res_x, det_y_svd + res_y

        dfp_points = {'det_x': det_x, 'det_y': det_y}
        return dfp_points

    @staticmethod
    def dfp_to_efp(transform, dfp_points, slice_phase=0.):
        """ Transform a point on the LMS detector (det_x, det_y) (w, efp_x, efp_y) onto the
        detector mosaic (det_x, det_y).  Note that the transform will normally have been verified as non-vignetting
        for this point in the EFP.
        slice_phase: Intra-slice position is degenerate with wavelength. It can be passed explicitly here, with
                     the default being to assume the centre of the slice in the EFP (slice_phase = 0.).
        """
        config = transform['configuration']
        ech_ord = config['ech_ord']
        slice_no = config['slice']
        det_x, det_y = dfp_points['det_x'], dfp_points['det_y']

        xp1, xp3, xp5 = config['xp1'], config['xp3'], config['xp5']
        yp1, yp3, yp5 = config['yp1'], config['yp3'], config['yp5']
        xp = [xp1, xp3, xp5]
        yp = [yp1, yp3, yp5]
        res_x = Util.offset_error_function(xp, det_x, 0.)
        res_y = Util.offset_error_function(yp, det_y, 0.)

        det_x_svd, det_y_svd = det_x - res_x, det_y - res_y
        matrices = transform['matrices']
        ai, bi = matrices['ai'], matrices['bi']
        phases, alphas = Util.apply_svd_distortion(det_x_svd, det_y_svd, ai, bi)

        ech_orders = np.full(alphas.shape, ech_ord)
        waves = Util.phases_to_waves(phases, ech_orders)

        efp_y_val = Util.slice_to_efp_y(slice_no, slice_phase)
        efp_y = np.full(alphas.shape, efp_y_val)

        efp_points = {'efp_x': alphas, 'efp_y': efp_y, 'efp_w': waves}
        return efp_points

    @staticmethod
    def is_det_hit(det_x, det_y):
        """ Check if a detector x, y coordinate (mm) falls on a light sensitive pixel.  Will be supreceded when
        we have a map from det_x, det_y to pix_col, pix_row
        """
        det_lims = Globals.det_lims
        for det_name in det_lims:
            x_lim, y_lim = det_lims[det_name]
            is_x_hit = x_lim[0] <= det_x and det_x < x_lim[1]
            is_y_hit = y_lim[0] <= det_y and det_y < y_lim[1]
            is_hit = is_x_hit and is_y_hit
            if is_hit:
                return  is_hit, det_name
        return False, None

    @staticmethod
    def offset_error_function(p, x, y):
        """ Polynomial with odd powered terms.
        """
        residual = (x * (p[0] + x * x * (p[1] + p[2] * x * x))) - y
        return residual

    @staticmethod
    def fit_residual(x_ref, x_in, y_in):
        _, xy_sort = Util.sort(x_ref, (x_in, y_in))
        guess_oc = [0.0, 0.0, 0.0]
        res_lsq = least_squares(Util.offset_error_function, guess_oc, args=xy_sort)
        offset_correction = res_lsq.x
        return offset_correction

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
        ech_order, slice_no, spifu_no = config
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
        mat_names = ["A", "B", "AI", "BI"]
        label = kwargs.get('label', '---')
        print(label)
        for matrix, mat_name in zip(transform, mat_names):
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
    def find_fwhm_lins(x, ys):
        """ Find fwhm for """
        ys = np.atleast_2d(ys)
        nr, nc = ys.shape
        fwhm_lin_list = []
        for r in range(0, nr):
            y = ys[r, :]
            xl, xr, yh = Util.find_fwhm_lin(x, y)
            fwhm_lin_list.append(xr - xl)
        fwhms_lin = np.array(fwhm_lin_list)
        fwhm_lin = np.mean(fwhms_lin)
        fwhm_lin_err = np.std(fwhms_lin) if nc > 1 else 0.
        return fwhm_lin, fwhm_lin_err

    @staticmethod
    def print_minmax(obs_list):
        print("{:>10s}{:>10s}{:>10s}".format('Obs No.', 'min', 'max'))
        for i, obs in enumerate(obs_list):
            image, param = obs
            amin, amax = np.amin(image), np.amax(image)
            fmt = "{:>10d}{:10.2e}{:10.2e}"
            print(fmt.format(i, amin, amax))
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
        # @author Tea Temim
        # Converted to python from Ronayette's original IDL code (Temim)
        # 21/2/17
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

        dim = order+1

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
        w_cutoff = 1.0e-5            # Only use singular values >= w_cutoff
        for k in range(0, n_w):
            if abs(w[k]) >= w_cutoff:
                wp[k, k] = 1.0 / w[k]
        a = xo @ u.T @ wp @ v.T     # (Was wp.T, but wp is square diagonal)
        amat = np.reshape(a, (dim, dim))
        b = yo @ u.T @ wp @ v.T
        bmat = np.reshape(b, (dim, dim))
        return amat, bmat

    @staticmethod
    def find_phase_bounds(det_bounds, tf_list):
        x, y = det_bounds
        for tf in tf_list:
            ech_order, slice_no, spifu_no, a, b, ai, bi = tf
            u, v = Util.apply_distortion(x, y, ai, bi)
        return u, v

    @staticmethod
    def make_rgb_gradient(vals):
        n_pts = len(vals)
        rgb = np.zeros((n_pts, 3))

        w_min = np.amin(vals)
        w_max = np.amax(vals)
        r_min, r_max = 0., 1.
        b_min, b_max = 1., 0.

        f = (vals - w_min) / (w_max - w_min)
        rgb[:, 0] = r_min + f * (r_max - r_min)
        rgb[:, 1] = np.sin(f * math.pi)
        rgb[:, 2] = b_min + f * (b_max - b_min)
        return rgb
