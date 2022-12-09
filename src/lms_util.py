#!/usr/bin/env python
"""
@author: Alistair Glasse

Update:
"""
import numpy as np
from lms_globals import Globals


class Util:

    def __init__(self):
        Util.n_lines_config = -1
        return

    @staticmethod
    def find_dispersion(transform, configuration):
        dw = 0.001  # Choose a wavelength interval of 1 nanometre (approx. 4000 / 50 = 80 pixels)

        slice, wave, prism_angle, grating_angle, order, im_pix_size = configuration
        a, b, ai, bi = transform

        k = order / (2.0 * Globals.rule_spacing)
        phases = [k * wave, k * (wave + dw)]
        alpha_middle = 0.0  # FP2 coordinate
        alphas = [alpha_middle, alpha_middle]
        det_x, det_y = Util.apply_distortion(phases, alphas, a, b)  # Detector coordinate in mm
        dx = det_x[1] - det_x[0]
        dw_dx = dw / dx  # micron per mm
        dw_lmspix = dw_dx * Globals.mm_lmspix  # micron per pixel
        return dw_lmspix

    @staticmethod
    def openw_transform_file(transform_file, n_terms):
        tf_file = open(transform_file, 'w')

        fmtA = "{:>6s},{:>3s},{:>6s},{:>8s},{:>8s},{:>8s},{:>8s},{:>3s},{:>3s},{:>3s},"
        fmtB = "{:>15s}," * n_terms       #{:>15s},{:>15s},{:>15s},{:>15s},"
        fmtC = "{:>8s},{:>8s},{:>8s},{:>8s}\n"
        hdr = fmtA.format('Ech', 'Ord', 'Prism', 'w1', 'w2', 'w3', 'w4', 'Sli', 'Mat', 'Row')
        tf_file.write(hdr)
        hdr = fmtB.format('Col_0', 'Col_1', 'Col_2', 'Col_3', 'Col_4')
        tf_file.write(hdr)
        hdr = fmtC.format('<dx>', 'std(dx)', '<<dy>', 'std(dy)')
        tf_file.write(hdr)
        hdr = fmtA.format('deg', ' ', 'deg', 'um', 'um', 'um', 'um', ' ', ' ', ' ')
        tf_file.write(hdr)
        hdr = fmtB.format('0', '1', '2', '3', '4')
        tf_file.write(hdr)
        hdr = fmtC.format('mm', 'mm', 'mm', 'mm')
        tf_file.write(hdr)
        return tf_file

    @staticmethod
    def read_transform_file(transform_file):
        """ Read in all zemax transform data and update the global variables
        to match.
        """
        fmt = "Reading transform list from file= {:s}"
        print(fmt.format(transform_file))
        lines = open(transform_file, 'r').read().splitlines()
        n_hdr_lines = 2
        tf_list = lines[n_hdr_lines:]
        is_mat_zero = True
        i = 0
        while is_mat_zero:
            tokens = tf_list[i].split(',')
            mat = int(tokens[8])
            is_mat_zero = mat == 0
            i = i + 1
        n_rows_mat = i - 1
        n_lines = len(tf_list)
        n_lines_mat = n_rows_mat
        n_lines_slice = n_lines_mat * Globals.n_mats_transform
        Util.n_lines_config = n_lines_slice * Globals.n_slices
        Globals.n_configs = int(n_lines / Util.n_lines_config)
        return tf_list

    def extract_configs(self, tf_list):
        """ Extract unique configuration data (echelle angle, wavelength etc.)
        from a transform list.
        :param tf_list:
        :return:
        """
        n_lines = len(tf_list)
        n_configs = int(n_lines / Util.n_lines_config)
        pas = np.zeros(n_configs)
        eas = np.zeros(n_configs)
        eos = np.zeros(n_configs)
        w1s = np.zeros(n_configs)
        w2s = np.zeros(n_configs)
        w3s = np.zeros(n_configs)
        w4s = np.zeros(n_configs)
        for i in range(0, n_configs):
            j = i * Util.n_lines_config
            tokens = tf_list[j].split(',')
            eas[i] = float(tokens[0])
            eos[i] = float(tokens[1])
            pas[i] = float(tokens[2])
            w1s[i] = float(tokens[3])
            w2s[i] = float(tokens[4])
            w3s[i] = float(tokens[5])
            w4s[i] = float(tokens[6])
        configs = (eas, eos, pas, w1s, w2s, w3s, w4s)
        return configs

    @staticmethod
    def find_polyfit(x, coeffs, **kwargs):
        order = kwargs.get('order', 1)
        ncr = coeffs.shape[0]
        poly = np.zeros((ncr, ncr, order+1))
        for i in range(0, ncr):
            for j in range(0, ncr):
                y = coeffs[i, j]
                poly[i, j] = np.polyfit(x, y, order)
        return poly

    def print_stats(self, st_file, data):
        fmt1 = "{:15s}{:15s}"
        fmt2 = "{:20s}{:20s}{:20s}"
        fmt3 = "{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}"
        if data is None:        # Write header only
            hdr = fmt1.format("Transform", "Polynomial")
            hdr = hdr + fmt2.format("Mean offset", "Std deviation", "Maximum")
            print(hdr)
            st_file.write(hdr + '\n')
            hdr = fmt1.format("matrix terms", "fit terms")
            hdr = hdr + fmt3.format("X", "Y", "X", "Y", "X", "Y")
            print(hdr)
            st_file.write(hdr + '\n')
        else:
            n_terms, poly_order, ox, oy = data
            mx = 1000.0 * np.mean(ox)
            my = 1000.0 * np.mean(oy)
            vx = 1000.0 * np.std(ox)
            vy = 1000.0 * np.std(oy)
            zx = 1000.0 * np.max(ox)
            zy = 1000.0 * np.max(oy)

            fmt4 = "{:10d}{:15d}"
            text = fmt4.format(n_terms, poly_order+1)
            fmt5 = "{:15.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}{:10.2f}"
            text = text + fmt5.format(mx, my, vx, vy, zx, zy)
            print(text)
            st_file.write(text + '\n')

    @staticmethod
    def filter_transform_list(list, n_terms, e_order, sno, mat_idx):
        """ Filter transform data with matching echelle order and slice number.
        """
        n_mat_transform = 4
        n_slices = 28

        n_lines = len(list)
        n_lines_mat = n_terms
        n_lines_transform = n_lines_mat * n_mat_transform
        n_lines_slice = n_lines_mat * n_mat_transform
        n_lines_config = n_lines_slice * n_slices
        n_transforms = int(n_lines / n_lines_transform)

        n_configs = int(n_lines / n_lines_config)

        # For maximum flexibility, read lines until the target echelle order
        # and slice are found.
        slice_idx = sno - 1

        n_config_pars = 8
        configs = np.zeros((n_config_pars, n_transforms))
        transforms = np.zeros((n_terms, n_terms, n_transforms))

        tr_idx = 0
        for start_line in range(0, n_lines, n_lines_transform):       # Search all transforms in list
            line = list[start_line]
            tokens = line.split(',')
            ech_order = int(tokens[1])
            s_idx = int(tokens[7])
            if e_order == ech_order and s_idx == slice_idx:
                for j in range(start_line, start_line + n_lines_transform, n_terms):
                    line = list[j]
                    tokens = line.split(',')
                    m_idx = int(tokens[8])
                    if m_idx == mat_idx:        # Read in filtered matrix
                        for i in range(0, n_config_pars):
                            configs[i, tr_idx] = tokens[i]
                        for r in range(0, n_terms):
                            line = list[j + r]
                            tokens = line.split(',')
                            for c in range(0, n_terms):
                                transforms[r, c, tr_idx] = tokens[c + 10]
                        tr_idx = tr_idx + 1
        return configs[:,0:tr_idx], transforms[:,:,0:tr_idx]

    def get_polyfit_transform(self, poly, ech_bounds, configuration):

        slice, _, _, grating_angle, order, im_pix_size = configuration
        e_idx = int(order - ech_bounds[0])
        shape = poly.shape
        n_mat = shape[2]
        n_terms = shape[4]
        tr = []

        for m in range(0, n_mat):
            mat = np.zeros((n_terms, n_terms))
            for i in range(0, n_terms):

                for j in range(0, n_terms):
                    poly_coeffs = poly[e_idx, slice, m, :, i, j]
                    mat[i, j] = np.polyval(poly_coeffs, grating_angle)
            tr.append(mat)
        return tr[0], tr[1], tr[2], tr[3]

    def read_polyfits_file(self, poly_file):
        pf = open(poly_file, 'r')
        lines = pf.read().splitlines()
        pf.close()
        tokens = lines[0].split(',')
        min_echelle_order = int(tokens[1])
        max_echelle_order = int(tokens[3])
        n_echelle_orders = max_echelle_order - min_echelle_order + 1
        n_slices = int(tokens[5])
        n_matrices = int(tokens[7])
        n_terms = int(tokens[9])
        poly_order = int(tokens[11])
        n_poly_terms = poly_order + 1
        poly = np.zeros((n_echelle_orders, n_slices, n_matrices, n_poly_terms, n_terms, n_terms))
        for line in lines[2:]:
            tokens = line.split(',')
            eo = int(tokens[0])
            s = int(tokens[1])
            mat = int(tokens[2])
            r = int(tokens[3])
            c = int(tokens[4])
            for i in range(0, n_poly_terms):
                poly[eo - min_echelle_order, s, mat, i, r, c] = float(tokens[5+i])
        return poly, (min_echelle_order, max_echelle_order)

    @staticmethod
    def _decode_tf_list_orders(tf_list):
        ech_orders = []
        for tf in tf_list:
            tokens = tf.split(',')
            eo = int(tokens[1])
            if eo not in ech_orders:
                ech_orders.append(eo)
        return ech_orders

    def write_polyfits_file(self, tf_list, n_terms, poly_order):

        poly_file = Globals.poly_file
        n_slices = Globals.n_slices
        n_mats = Globals.n_mats_transform
        pf = open(poly_file, 'w')
        ea_index = 0
        ech_orders = Util._decode_tf_list_orders(tf_list)
        fmt = "min_ech_order,{:3d},max_ech_order,{:3d},n_slices,{:3d},n_matrices_transform,{:4d},n_terms,{:4d},poly_order,{:4d}"
        hdr = fmt.format(min(ech_orders), max(ech_orders), n_slices, n_mats, n_terms, poly_order)
        pf.write(hdr + '\n')
        fmt = "{:3s},{:3s},{:3s},{:5s},{:3s},{:s}"
        hdr = fmt.format("eo", "sno", "mat", "row", "col", "poly_coeffs")
        pf.write(hdr + '\n')
        for eo in ech_orders:
            eo_token = "{:3d},".format(eo)
            for s in range(0, n_slices):
                s_token = "{:3d},".format(s)
                sno = s + 1
                for mat in range(0, n_mats):  # Process A B AI BI in one loop
                    m_token = "{:3d},".format(mat)
                    (configs, transforms) = self.filter_transform_list(tf_list, n_terms, eo, sno, mat)
                    xfit = configs[ea_index]
                    poly_fit = self.find_polyfit(xfit, transforms, order=poly_order)
#                    if eo == 23 and sno == 10:
#                        plot.config_v_coeffs(eo, sno, mat, xfit, transforms, poly_fit, suppress=False)
                    for i in range(0, n_terms):
                        for j in range(0, n_terms):
                            line = eo_token + s_token + m_token + "{:5d},{:3d},".format(i, j)
                            for k in range(0, poly_order + 1):
                                line = line + "{:15.7e},".format(poly_fit[i, j, k])
                            pf.write(line + "\n")
        pf.close()

    def print_poly_transform(self, transform, **kwargs):
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

    def distortion_fit(self, x_in, y_in, x_out, y_out, order, inverse=True):
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
        pow = np.array([range(0, dim), ] * x_out.size).transpose()

        mx = np.tile(np.power(xi, pow), (dim, 1))
        my = np.repeat(np.power(yi, pow), dim, axis=0)
        mxy = np.multiply(mx, my)
        m = np.multiply(mxy, mask)

        v, w, u = np.linalg.svd(m, full_matrices=False)
        n_w = w.size
        wp = np.zeros((n_w, n_w))
        w_cutoff = 1.0e-5            # Only use singular values >= w_cutoff
        for k in range(0, n_w):
            if (abs(w[k]) >= w_cutoff): wp[k,k]=1.0/w[k]
        a = xo @ u.T @ wp @ v.T     # (Was wp.T, but wp is square diagonal)
        amat = np.reshape(a, ((dim,dim)))
        b = yo @ u.T @ wp @ v.T
        bmat = np.reshape(b, ((dim,dim)))
        return amat, bmat

    @staticmethod
    def apply_distortion(x,y, a,b):
        """
        @author Alistair Glasse
        21/2/17   Create to encapsulate distortion transforms
        Apply a polynomial transform pair (A,B or AI,BI) to an array of points
        affine = True  Apply an affine version of the transforms (remove all non-
        linear terms)
        """

        dim = a.shape[0]
        n_pts = len(x)

        pow = np.array([range(0, dim), ] * n_pts).transpose()

        xmat = np.array([x, ] * dim)
        xin = np.power(xmat, pow)
        ymat = np.array([y, ] * dim)
        yin = np.power(ymat, pow)

        xout = np.zeros(n_pts)
        yout = np.zeros(n_pts)
        for i in range(0, n_pts):
            xout[i] = yin[:, i] @ a @ xin[:, i]
            yout[i] = yin[:, i] @ b @ xin[:, i]
        return xout, yout
