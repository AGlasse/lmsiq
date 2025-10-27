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
from lms_filer import Filer
from lmsdist_polyfit import PolyFit
from lms_globals import Globals


class DistortionMachine:
    """ Class containing methods for using LMS distortion transforms to map between the LMS entrance focal plane
    (wavelength, alpha, beta) and the detector focal plane (row, column).
    """

    def __init__(self):
        return

    def wab_to_det(self, wave, alpha, beta, opticon):
        filer = Filer('distortion', opticon)
        wpa_fit, wxo_fit, term_fits = filer.read_fit_parameters(opticon)
        lms_configs = PolyFit.wave_to_config(wave.value, wpa_fit, wxo_fit,
                                             debug=False, select='min_ech_ang', print_header=True)
        svd_transforms = PolyFit.make_slice_transforms(lms_configs[0], term_fits)



        return


