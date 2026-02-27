#!/usr/bin/env python
"""
@author: Alistair Glasse

Update:
"""
import math
import numpy as np
from scipy.optimize import curve_fit, fsolve
from lms_globals import Globals
from lmsdist_util import Util
from lmsdist_plot import Plot
from lms_transform import Transform


class PolyFit:
    """ PolyFit -
        Create interpolation parameters to map wavelength and transform matrix terms to echelle and prism angle
        settings.
    """
    # wpa_fit_order = None
    n_lines_config = -1

    def __init__(self, opticon):
        # PolyFit.wpa_fit_order = Globals.wpa_fit_order_dict[opticon]
        return

    @staticmethod
    def wave_to_config(wave, opticon, wpa_fit, wxo_fit, **kwargs):
        """ Find the LMS configuration (prism and echelle angles) which map a wavelength onto the centre of the
        detector mosaic, in the echelle diffraction order which has the echelle angle closest to zero.  The function
        is solved using the scipy 'fsolve' method.

        :param wave:  target wavelength to place at centre of mosaic
        :param wxo_fit: fit coefficients for wave x order surface as function of prism and echelle angles.
        :param term_fits: fit coefficients for terms in distortion matrices
        :return: (pa, ea) tuple containing optimum prism and echelle angles.
        """
        @staticmethod
        def ea_func(ea, *fargs):
            pa, wxo, fcoeff, debug = fargs
            xy = pa, ea[0]
            f = PolyFit.surface_model(xy, *fcoeff)
            if debug:
                print("{:>52.3f},{:>20.3f} -> dw = {:10.3f}".format(pa, ea[0], f - wxo))
            return f - wxo

        debug = kwargs.get('debug', False)
        select = kwargs.get('select', 'min_ech_ang')
        print_header = kwargs.get('print_header', True)

        lms_configs = []

        wpa_coeffs = wpa_fit['wpa_opt']
        pri_ang = PolyFit.poly_model(wave, *wpa_coeffs)

        # print("Finding configuration for wavelength = {:10.3f}".format(wave))
        wxo_coeffs = wxo_fit['wxo_opt']
        wxo_lim = np.array([108., 115.])      # Wavelength bounds of w x order coverage (in microns)
        ech_ang_min = 100.
        lms_config = Globals.lms_config_template.copy()
        for ech_ord in range(21, 42):
            lms_config['wave'] = wave
            lms_config['opticon'] = opticon

            wxo_tgt = wave * ech_ord
            if wxo_tgt < wxo_lim[0] or wxo_tgt > wxo_lim[1]:   # The accessible wavelength coverage in each order is wxo_lim / n
                continue
            ech_ang_guess = np.zeros(1)
            args = pri_ang, wxo_tgt, wxo_coeffs, debug
            ech_ang_sol, _, ier, msg = fsolve(ea_func, ech_ang_guess, args=args, full_output=True)
            ech_ang = ech_ang_sol[0]

            if select == 'all':
                lms_configs.append(lms_config)
                lms_config = Globals.lms_config_template.copy()
            if select == 'min_ech_ang':
                abs_ech_ang = math.fabs(ech_ang)
                if abs_ech_ang < ech_ang_min:
                    ech_ang_min = abs_ech_ang
                    lms_config['pri_ang'] = pri_ang
                    lms_config['ech_ang'] = ech_ang
                    lms_config['ech_order'] = ech_ord
        if select == 'all':
            return lms_configs
        return lms_config

    @staticmethod
    def create_polynomial_surface_fits(opticon, svd_transforms, plot_terms=False, plot_wxo=True):
        """ Calculate the best fit polynomials to map from wavelength x diffractionorder to echelle angle (wxo_fit),
        and all terms in the A, B, AI, BI transforms for all slices (term_fits).
        """
        slice_fits = {}
        term_fits = []
        wxo_fit, wxo_fit_order = None, None
        surface_model = PolyFit.surface_model
        for slice_no in Globals.slice_no_ranges[opticon]:
            for spifu_no in Globals.spifu_no_ranges[opticon]:
                slice_transforms = Util.filter_transform_list(svd_transforms,
                                                              slice_no=slice_no,
                                                              spifu_no=spifu_no)
                term_values = Util.get_term_values(slice_transforms, slice_no, spifu_no)
                term_fit = PolyFit.find_slice_fit(term_values)
                baseline_debug_level = Globals.debug_level
                if Globals.is_debug('medium'):
                    Plot.transform_fit(term_fit, term_values, surface_model, do_plots=True)
                    Plot.transform_fit(term_fit, term_values, surface_model, do_plots=True, plot_residuals=True)
                    Globals.set_debug_level(baseline_debug_level)
                term_fits.append([slice_no, spifu_no, term_fit])
                slice_fits[slice_no] = term_fit
                if slice_no == 13:
                    wxo_fit = PolyFit.find_wxo_fit(term_values, Globals.surface_fit_order)
                    if Globals.is_debug('medium'):
                        Plot.wxo_fit(wxo_fit, term_values, surface_model, plot_residuals=False)
                        Plot.wxo_fit(wxo_fit, term_values, surface_model, plot_residuals=True)
                        Globals.set_debug_level(baseline_debug_level)
        wxo_header = PolyFit.make_surface_model_header(wxo_fit['order'])
        return wxo_fit, wxo_header, term_fits

    @staticmethod
    def make_surface_model_header(order):
        ij = 0
        header = []
        for j in range(0, order):
            for i in range(0, order - j):
                text = "PA{:d}_EA{:d}".format(i, j)
                header.append(text)
                ij += 1
        return header

    @staticmethod
    def surface_model(xy, *coeff):
        """ Fit function for a two variable quadratic surface of the form,
         f(x, y) =  a00     + a01 x  + a02 x^2 + a03 x^2
                    a10 y   + a11 xy + a12 x^2y
                    a20 y^2 + a20 xy^2
                    a30 y^3
         for example, x -> prism angle, y -> echelle angle, f(x, y) -> wavelength (or transform coefficient)
        """
        xp, ye = xy
        ij = 0
        f = 0.
        order = Globals.surface_fit_order
        for j in range(0, order):
            y_term = np.power(ye, j)
            for i in range(0, order - j):
                x_term = np.power(xp, i)
                f += coeff[ij] * x_term * y_term
                ij += 1
        return f

    @staticmethod
    def poly_model(x, *coeff):
        """ Evaluate a polynomial function of the form y = c0 + c1 x + c2 x x + ..
        """
        f = 0.
        order = len(coeff)
        for i in range(0, order):
            f += coeff[i] * np.power(x, i)
        return f

    @staticmethod
    def create_polynomial_fit(waves, pri_angs, poly_order=3):
        """ Calculate the best fit polynomial to map from wavelength to prism angle
        """
        p0 = [0] * poly_order
        wpa_opt, wpa_cov = curve_fit(PolyFit.poly_model, xdata=waves, ydata=pri_angs, p0=p0)
        wpa_fit = {'wpa_opt': wpa_opt, 'wpa_cov': wpa_cov, 'n_coeffs': poly_order}
        return wpa_fit

    @staticmethod
    def find_slice_fit(term_vals):
        n_coefficients = Globals.surface_fit_n_coeffs     # No. of fit coefficients (upper triangular elements)

        x = np.array(term_vals['pri_ang'])
        y = np.array(term_vals['ech_ang'])
        matrices = term_vals['matrices']
        slice_fit = {}
        mat_tags = ['a', 'b', 'ai', 'bi']
        for i, mat_tag in enumerate(mat_tags):
            term_array = np.zeros((4, 4, n_coefficients))
            slice_fit[mat_tag] = term_array
            for row in range(0, 4):
                for col in range(0, 4):
                    term_list = []
                    for mat_set in matrices:
                        mat = mat_set[mat_tag]
                        term = mat[row, col]
                        term_list.append(term)
                    terms = np.array(term_list)
                    term_opt, term_cov = curve_fit(PolyFit.surface_model,
                                                   xdata=(x, y), ydata=terms, p0=[0]*n_coefficients)
                    term_array[row, col, :] = term_opt
        return slice_fit

    @staticmethod
    def find_wxo_fit(term_values, order):
        n_coefficients = Globals.surface_fit_n_coeffs
        slice_no = term_values['slice_no']
        spifu_no = term_values['spifu_no']
        x = np.array(term_values['pri_ang'])
        y = np.array(term_values['ech_ang'])
        wxo = np.array(term_values['w_bs']) * np.array(term_values['ech_ords'])
        wxo_opt, wxo_cov = curve_fit(PolyFit.surface_model,
                                     xdata=(x, y), ydata=list(wxo), p0=[0]*n_coefficients)
        wxo_fit = {'slice_no': slice_no, 'spifu_no': spifu_no,
                   'wxo_opt': wxo_opt, 'wxo_cov': wxo_cov,
                   'order': order, 'n_coefficients': n_coefficients}
        return wxo_fit

    @staticmethod
    def make_fit_matrix(poly_matrix, pa, ea, eo):
        """ Create a transform matrix at any prism and echelle angle with terms provided using the
        2nd order 2D polynomial fit function.

        :param opt_mat:
        :param pa:
        :param ea:
        :param eo:
        :return:
        """
        svd_order = Globals.svd_order
        svd_shape = svd_order, svd_order
        matrix = np.zeros(svd_shape)
        for i in range(0, svd_order):
            for j in range(0, svd_order):
                fit_terms = list(poly_matrix[i, j])
                matrix[i, j] = PolyFit.surface_model((pa, ea), *fit_terms)
        return matrix

    # @staticmethod
    # def make_slice_transforms(lms_config, term_fits):
    #     slice_transforms = []
    #     for term_fit in term_fits:
    #
    #
    #
    #
    #
    #
    #     return slice_transforms

    @staticmethod
    def make_fit_transform(cfg, term_fit):
        pa, ea, eo = cfg['pri_ang'], cfg['ech_ang'], cfg['ech_order']
        matrices = {}
        for mat_name in Globals.matrix_names:
            poly_matrix = term_fit[mat_name]
            matrix = PolyFit.make_fit_matrix(poly_matrix, pa, ea, eo)
            matrices[mat_name] = matrix
        transform = Transform(cfg=cfg, matrices=matrices)
        return transform

    @staticmethod
    def make_mfp_projection(slice_no, spifu_no, trace, svd_transform, wxo_fit, term_fits):

        svd_cfg = svd_transform.configuration
        svd_slice_config = svd_transform.slice_configuration
        slice_no = svd_slice_config['slice_no']

        term_fit = term_fits[slice_no][spifu_no]
        fit_transform = PolyFit.make_fit_transform(svd_cfg, term_fit)

        slice_id = {'slice_no': slice_no, 'spifu_no': spifu_no}
        efp_x = trace.get('efp_x', **slice_id)
        efp_y = trace.get('efp_y', **slice_id)
        efp_w = trace.get('wavelength', **slice_id)
        efp_points = {'efp_y': efp_y, 'efp_x': efp_x, 'efp_w': efp_w}  # Centre of slice for detector
        mfp_points_zemax = {'mfp_x': trace.get('det_x', **slice_id),
                            'mfp_y': trace.get('det_y', **slice_id)}
        mfp_points_svd = Util.efp_to_mfp(svd_transform, efp_points)
        mfp_points_fit = Util.efp_to_mfp(fit_transform, efp_points)
        projection = {'zemax': mfp_points_zemax, 'svd': mfp_points_svd[0], 'fit': mfp_points_fit[0]}
        return projection
