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

    def wab_to_transform(self, configs, opticon):
        """ Find the set of transform matrices (A, B, AI, BI, M, MI) which map a specific EFP alpha, beta, wave coordinate onto
        the detector focal plane.
        """
        # lms_config, slice_config = configs
        filer = Filer()
        filer.set_configuration('distortion', opticon)
        fit_data = filer.read_fit_parameters(opticon)
        svd_transform = PolyFit.make_fit_transform(configs, fit_data)
        model_config = Globals.model_configurations['distortion'][opticon]
        date_stamp = model_config[2]
        affines = filer.read_fits_affine_transform(date_stamp)
        return svd_transform, affines
