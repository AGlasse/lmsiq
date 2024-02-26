#!/usr/bin/env python
"""
@author: Alistair Glasse

Update:
"""
import numpy as np
from lms_filer import Filer
from lms_globals import Globals


class Wcal:

    fit_order = 1
    p_atow, p_wtoa = None, None

    def __init__(self):
        return

    @staticmethod
    def find_dispersion(traces, obs_dict):
        """ Find the dispersion (micron per pixel) for the LMS configuration defined in
        obs_dict using the nearest transform in 'traces'
        """
        dw_lms_pix = 1.0
        optical_configuration = obs_dict['optical_configuration']
        date_stamp = obs_dict['optical_configuration']
        model_configuration = 'distortion', optical_configuration, date_stamp
        dist_filer = Filer(model_configuration)
        traces = dist_filer.read_pickle(dist_filer.trace_file)

        # if is_spifu:
        #     traces = Filer.



        return dw_lms_pix

    @staticmethod
    def write_poly(wcal_file, configs, wave_limits):
        """ Write the polynomial coefficients which transform between wavelength
        and echelle angle to a .csv file.
        """
        eos, eas, pas, w1s, w2s, w3s, w4s = configs[0:7]
        w1s, w2s, w3s, w4s = wave_limits
        eomin = int(np.min(eos))
        eomax = int(np.max(eos))
        n_eos = eomax - eomin + 1
        wbs = (w2s + w3s) / 2.0

        f = open(wcal_file, 'w')
        fmt = "{:>15s},{:>15s},{:>15s},{:>15s},{:>15s},\n"
        f.write(fmt.format("Echelle Order", "Poly Coeffs.", "wav->ang", "Poly Coeffs.", "ang->wav"))
        for i in range(0, n_eos):
            eo = eomin + i
            line = "{:15d},".format(eo)
            idx = np.where(eos == eo)
            w, a = wbs[idx], eas[idx]
            p_wtoa = np.poly1d(np.polyfit(w, a, Wcal.fit_order))        # Wavelength to angle
            for term in p_wtoa:
                token = "{:15.7e},".format(term)
                line = line + token
            p_atow = np.poly1d(np.polyfit(a, w, Wcal.fit_order))        # Angle to wavelength
            for term in p_atow:
                token = "{:15.7e},".format(term)
                line = line + token
            f.write(line + "\n")
        f.close()
        return

    @staticmethod
    def read_poly(wcal_file):
        """ Read the polynomial coefficients which transform between wavelength
        and echelle angle from a .csv file.
        """
        fmt = "Reading echelle angle v wavelength polynomials from file= {:s}"
        print(fmt.format(wcal_file))
        lines = open(wcal_file, 'r').read().splitlines()
        n_hdr_lines = 1
        efw_list = lines[n_hdr_lines:]
        n_eos = len(efw_list)
        n_terms = Wcal.fit_order + 1
        p_wtoa, p_atow = np.zeros((n_eos, n_terms + 1)), np.zeros((n_eos, n_terms + 1))

        i = 0
        for rec in efw_list:
            tokens = rec.split(',')
            p_wtoa[i, 0] = int(tokens[0])
            p_atow[i, 0] = int(tokens[0])
            for j in range(0, n_terms):
                p_wtoa[i, 1+j] = float(tokens[1+j])
                p_atow[i, 1+j] = float(tokens[1+j+n_terms])
            i = i + 1
        Wcal.p_wtoa = p_wtoa
        Wcal.p_atow = p_atow
        return

    @staticmethod
    def find_wavelength(angle, order):
        """ Calculate the wavelength corresponding to an echelle order and
        angle of incidence using the polynomial transform. """
        p_atow = Wcal.p_atow
        i = int(order - p_atow[0, 0])
        p = p_atow[i, 1:Wcal.fit_order + 2]
        w = np.polyval(p, angle)
        return w

    @staticmethod
    def get_polyfit_transform(poly, ech_bounds, configuration):

        slice_id, _, _, grating_angle, order, im_pix_size = configuration
        e_idx = int(order - ech_bounds[0])
        shape = poly.shape
        n_mat = shape[2]
        n_terms = shape[4]
        tr = []

        for m in range(0, n_mat):
            mat = np.zeros((n_terms, n_terms))
            for i in range(0, n_terms):

                for j in range(0, n_terms):
                    poly_coeffs = poly[e_idx, slice_id, m, :, i, j]
                    mat[i, j] = np.polyval(poly_coeffs, grating_angle)
            tr.append(mat)
        return tr[0], tr[1], tr[2], tr[3]

    @staticmethod
    def find_echelle_angle(w, order):
        """ Cacluate the echelle angle of incidence as a function of wavelength (micron)
        and diffraction order number. """
        p_wtoa = Wcal.p_wtoa
        i = int(order - p_wtoa[0, 0])
        p = p_wtoa[i, 1:Wcal.fit_order + 2]
        ea = np.polyval(p, w)
        return ea

    @staticmethod
    def find_echelle_setting(w):
        """ Find the optimum order and echelle angle for observing this
        wavelength """
        p_wtoa = Wcal.p_wtoa
        n_orders = len(p_wtoa)
        eo_opt = -1
        ea_min = 90.0
        for i in range(0, n_orders):
            eo = p_wtoa[i, 0]
            p = p_wtoa[i, 1:Wcal.fit_order+2]
            ea = np.polyval(p, w)
            if abs(ea) < abs(ea_min):
                ea_min = ea
                eo_opt = eo
        return int(eo_opt), ea_min
