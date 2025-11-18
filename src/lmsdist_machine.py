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
        config = Globals.model_configurations[opticon]
        date_stamp = config[2]
        affines = Filer.read_fits_affine_transform(date_stamp)


        # For the polynomial fits, start with the fit transforms corresponding to the trace configuration.
        # lms_config = trace.lms_config
        # lms_fit_config = PolyFit.wave_to_config(wave_bs, opticon, wpa_fit, wxo_fit)

        # pri_ang_bs = PolyFit.wpa_model(wave_bs, *wpa_fit['opt'])
        # We now need to find the echelle angle which maps to the wavelength x order product for this prism angle.
        # To start with this is done using crude recursion.  We expect that the wxo
        # ech_order = lms_config['ech_order']
        # ech_ang_fit = -100.
        # hunting = True
        # ea, ea_inc = 0., 0.01
        # wxo_tgt = wave_bs * ech_order
        # err_tgt= 0.01
        # dea_dwxo = -12. / 5.       # wxo changes by -5 microns over 12 degrees echelle rotation.
        # while hunting:
        #     wxo = PolyFit.surface_model((pri_ang_bs, ea), *wxo_fit['opt'])
        #     wxo_err = wxo - wxo_tgt
        #     if abs(wxo_err) < err_tgt:
        #         ech_ang_fit = ea
        #         hunting = False
        #     else:
        #         sign_err = math.copysign(1, wxo_err)
        #         ea_inc = 0.5 * wxo_err * dea_dwxo
        #         ea -= ea_inc
        # pri_ang_fit = pri_ang_bs
        # cfg = {'pri_ang': pri_ang_fit, 'ech_ang': ech_ang_fit, 'ech_order': ech_order}




        return


