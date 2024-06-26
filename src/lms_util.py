#!/usr/bin/env python
"""
@author: Alistair Glasse

Update:
"""
import numpy as np
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
                det_x, _ = trace.apply_polynomial_distortion(np.array(phases), np.array(alphas), a, b)
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
    def efp_to_dfp(traces, efp_point, tgt_config):
        """ Transform a point in the LMS entrance focal plane (w, efp_x, efp_y) onto the
        detector mosaic (det_x, det_y), given an LMS mechanism configuration by interpolation of
        nearby points.
        For the nominal optical path
        1. Find the IFU slice corresponding to (efp_x, efp_y).
        2.
        """
        efp_x, efp_y, wave = efp_point
        tgt_ech_order, tgt_slice_no, tgt_spifu_no = tgt_config
        # Find the 4 nearest neighbour traces
        nn_slice_objects = []
        for trace in traces:
            for slice_object in trace.slice_objects:
                config, transforms, corr_polys, rays, wave_bounds = slice_object
                order, slice_no, spifu_no = config
                # print(order, slice_no, spifu_no)
                is_order = order == tgt_ech_order
                if not is_order:
                    continue
                is_slice_no = slice_no == tgt_slice_no
                if not is_slice_no:
                    continue
                is_spifu_no = spifu_no == tgt_spifu_no
                if not is_spifu_no:
                    continue
                nn_slice_objects.append(slice_object)
        # Loop through nearest neighbour transforms
        for nn_so in nn_slice_objects:
            nn_config, nn_transforms, nn_corr_polys, nn_rays, _ = nn_so
        return 0., 0.

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
    def distortion_fit(x_in, y_in, x_out, y_out, order, inverse=True):
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
