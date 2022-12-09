#!/usr/bin/env python
"""
@author: Alistair Glasse

Update:
"""
import math
import numpy as np
from lms_wcal import Wcal

class Efficiency:


    def __init__(self):
        return

    @staticmethod
    def combined(wave, ech_order):
        eff_slow = Efficiency.slow(wave)
        eff_blaze = Efficiency.blaze(wave, ech_order)
        eff_combi = eff_slow * eff_blaze
        return eff_combi

    @staticmethod
    def slow(wave):
        r_mirror = 0.99         # Mirror reflectivity
        n_tfer = 4              # Transfer optics reflections
        n_ifu = 4               # IFU reflections
        n_pre = 5           # Pre-disperser reflections
        n_main = 6              # Main disperser reflections
        n_mirrors = n_tfer + n_ifu + n_pre + n_main
        t_mirrors = math.pow(r_mirror, n_mirrors)
        t_znse = 0.98          # Transmission of AT coated ZnSe prism
        n_prism = 4             # No of prism surfaces transited
        t_prism = math.pow(t_znse, n_prism)
        t_slow = t_mirrors * t_prism
        return t_slow

    @staticmethod
    def blaze(wave, order):
        """ Efficiency model;  eff = eff_max cos (k x echelle_angle)
        :param wave:
        :return:
        """
        eff_max = 0.75
        eff_width = 0.23            # Incidence angle where efficiency of adjacent orders is equal

        ech_angle = Wcal.find_echelle_angle(wave, order)
        ea = ech_angle * math.pi / 180.0
        phase = order * math.pi * math.sin(ea) * eff_width
        amp = math.sin(phase) / phase
        eff = eff_max * amp * amp
        return eff

    @staticmethod
    def test(w_test_lim):
        """ Method to generate array of efficiency values. """
        omax, amax = Wcal.find_echelle_setting(w_test_lim[0])
        omin, amin = Wcal.find_echelle_setting(w_test_lim[1])

        n_orders = int(omax - omin)
        amin = -6.0
        amax = 6.0
        n_pts = 50
        da = (amax - amin) / (n_pts - 1)
        orders = np.arange(omin, omax, 1)
        angs = np.zeros(n_pts)
        wavs, effs, = np.zeros((n_pts, n_orders)), np.zeros((n_pts, n_orders))
        for i in range(0, n_pts):
            ang = amin + i * da
            angs[i] = ang
            for j in range(0, n_orders):
                order = orders[j]
                wav = Wcal.find_wavelength(ang, order)
                slow_eff = Efficiency.slow(wav)
                blaze_eff = Efficiency.blaze(wav, order)
                wavs[i,j] = wav
                effs[i,j] = blaze_eff   # * slow_eff
        return wavs, effs, orders, angs
