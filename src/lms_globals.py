#!/usr/bin/env python
"""
@author Alistair Glasse
Python object to encapsulate LMS optical constants

29/7/19  Created class (Glasse)
"""


class Globals:

    efp_as_mm = 0.303           # Plate scale (arcsec/mm) in entrance focal plane
    alpha_fov = 1.0             # Set up field of view (arcsec)
    beta_fov = 0.5
    rule_spacing = 18.2			# Echelle rule spacing [um]
    n_slices = 28
    n_mats_transform = 4        # Four matrices per transform (A, B, AI, BI)
    mm_lmspix = 0.018
#    mm_fitspix = -1.0           # Size of 'Proper' image pixel, read from fits file header in lmsiq_analyse.lsf
    spatial_scale = 0.0082      # Along slice arcsec / pixel
    pix_edge = 2048             # H2RG format
    det_gap = 3.0				# Gap between active regions of detectors in 2 x 2 mosaic (mm)
    pix_margin = [64, 64]		# Unilluminated margin around all detector (pixels)
    margin = pix_margin[0] * mm_lmspix
    n_configs = -1              # Number of configurations, calculated when lms_dist_buffer is read.
    # File locations and names
    zem_folder = '../input_zemax/'
    trace_folder = '../output/traces/'
    transform_file = '../output/lms_dist_buffer.txt'
    poly_file = '../output/lms_dist_poly_old.txt'
    wcal_file = '../output/lms_dist_wcal.txt'             # Echelle angle as function of wavelength
    stats_file = '../output/lms_dist_stats.txt'
