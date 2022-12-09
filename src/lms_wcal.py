#!/usr/bin/env python
"""
@author: Alistair Glasse

Update:
"""
import numpy as np
from lms_globals import Globals


class Wcal:

    fit_order = 1

    def __init__(self):
        return

    @staticmethod
    def write_poly(configs):
        """ Write the polynomial coefficients which transform between wavelength
        and echelle angle to a .csv file.
        """
        efw_file = Globals.wcal_file
        (eas, eos, pas, w1s, w2s, w3s, w4s) = configs
        eomin = int(np.min(eos))
        eomax = int(np.max(eos))
        n_eos = eomax - eomin + 1
        wbs = (w2s + w3s) / 2.0

        f = open(efw_file, 'w')
        fmt = "{:>15s},{:>15s},{:>15s},{:>15s},{:>15s},\n"
        f.write(fmt.format("Echelle Order", "Poly Coeffs.", "wav->ang", "Poly Coeffs.", "ang->wav"))
        for i in range(0, n_eos):
            eo = eomin + i
            line = "{:15d},".format(eo)
            idx = np.where(eos == eo)
            p_wtoa = np.poly1d(np.polyfit(wbs[idx], eas[idx], Wcal.fit_order))        # Wavelength to angle
            for term in p_wtoa:
                token = "{:15.7e},".format(term)
                line = line + token
            p_atow = np.poly1d(np.polyfit(eas[idx], wbs[idx], Wcal.fit_order))        # Angle to wavelength
            for term in p_atow:
                token = "{:15.7e},".format(term)
                line = line + token
            f.write(line + "\n")
        f.close()
        return

    @staticmethod
    def read_poly():
        """ Read the polynomial coefficients which transform between wavelength
        and echelle angle from a .csv file.
        """
        wcal_file = Globals.wcal_file
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
