#!/usr/bin/env python
"""
@author Alistair Glasse
Python object to encapsulate LMS optical constants

29/7/19  Created class (Glasse)
"""


class Globals:
    zemax_configuration = None
    det_pix_size, im_pix_size = None, None
    # Optical configurations
    nominal = 'nominal'
    spifu = 'spifu'
    optical_configurations = [nominal, spifu]

    ipc_on_tag, ipc_off_tag = '_ipc_01_3', '_ipc_00_0'      # IPC/diffusion file tags
    efp_as_mm = 0.303           # Plate scale (arcsec/mm) in entrance focal plane
    alpha_fov = 1.0             # Set up field of view (arcsec)
    beta_fov = 0.5
    alpha_mas_pix = 8.7         #
    beta_mas_pix = 20.7
    rule_spacing = 18.2			# Echelle rule spacing [um]
    ge_refracive_index = 4.05   # Refractive index of germanium
    n_lms_slices = 28
    n_lms_spifu_slices = 6
    n_mats_transform = 4        # Four matrices per transform (A, B, AI, BI)
    nom_pix_pitch = 18.0        # LMS pixel pitch in microns
    spatial_scale = 0.0082      # Along slice arcsec / pixel
    det_gap = 3.0				# Gap between active regions of detectors in 2 x 2 mosaic (mm)
    pix_margin = [64, 64]		# Unilluminated margin around all detector (pixels)
    margin = pix_margin[0] * nom_pix_pitch / 1000.0     # Convert to mm
    n_configs = -1              # Number of configurations, calculated when lms_dist_buffer is read.
    # Image quality parameters are calculated for three levels of data product
    process_levels = ['raw_zemax', 'proc_zemax', 'proc_detector']
    axes = ['spectral', 'spatial']

    # @staticmethod
    # def get_im_oversampling(process_level):
    #     im_oversampling = Globals.det_pix_size / Globals.im_pix_size
    #     if process_level == 'proc_detector':
    #         im_oversampling = 1.
    #     return int(im_oversampling)
