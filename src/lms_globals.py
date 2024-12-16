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
    slice_no_ranges = {nominal: range(1, 29), spifu: range(12, 15)}
    spifu_no_ranges = {nominal: range(0, 1), spifu: range(1, 7)}

    ipc_on_tag, ipc_off_tag = '_ipc_01_3', '_ipc_00_0'      # IPC/diffusion file tags

    slice_id_fmt = "{:s}_{:d}_{:d}_{:02d}_{:d}"        # Define transforms by opticon, ech_ord, slice_no, spifu_no

    # Plate scale at the entrance focal plane.  Defined in LB email 14/11/24 (in ../docs/lb_ps_131124.txt)
    # as efp_as_mm = 350.06 / 1937 = 0.180723
    efp_as_mm = 0.180723

    # Plate scale at detector
    alpha_mas_pix = 8.7                                     #
    beta_mas_pix = 20.7
    # The field of view in the optical design is quoted in the FDR design report (E-REP-ATC-MET-1003) is then
    alpha_fov_as = 0.897
    beta_fov_as = beta_mas_pix * 28 / 1000.
    efp_x_fov_mm = alpha_fov_as / efp_as_mm   # EFP field of view (mm) (Note ray trace bounds 5.842063, 3.208105)
    efp_y_fov_mm = beta_fov_as / efp_as_mm

    debug = False
    if debug:
        fmt = "{:>30s} = {:5.3f} x {:5.3f} {:s}"
        print(fmt.format('EFP field of view, alpha, beta', efp_x_fov_mm, efp_y_fov_mm, 'mm'))
        print(fmt.format('', alpha_fov_as, beta_fov_as, 'arcseconds'))

    # Diffraction grating parameters
    blaze_angle = 51.23                                     # Echelle blaze angle (deg)
    rule_spacing = 18.2			                            # Echelle rule spacing [um]
    ge_refractive_index = 4.05                              # Refractive index of germanium
    wav_first_order = 2. * rule_spacing * ge_refractive_index
    wav_first_order = 21 * 5.216                            # First order blaze wavelength

    # IFU parameters
    n_lms_slices = 28
    n_lms_spifu_slices = 6

    transform_config = {'n_mats': 4,                        # Four matrices per transform (A, B, AI, BI)
                        'mat_order': 4,                     # No of rows/columns in A, B, AI, BI
                        'res_fit_order': 3}                 # No. of terms in residual fit
    svd_cutoff = 1.0e-7                                     # SVD eigenvalues below this value set to zero.
    nom_pix_pitch = 18.0                                    # LMS pixel pitch in microns
    # efp_alpha_size = alpha_fov * nom_pix_pitch / alpha_mas_pix      # EFP along-slice in mm
    # spatial_scale = 0.0082                                  # Along slice arcsec / pixel
    det_gap = 3.0				            # Gap between active regions of detectors in 2 x 2 mosaic (mm)
    pix_margin = [64, 64]		            # Unilluminated margin around outer detector edge (pixels)
    margin = pix_margin[0] * nom_pix_pitch / 1000.          # Convert to mm
    det_size = 2048 * nom_pix_pitch / 1000.     # Detector size in mm
    xyn = .5 * det_gap                      # x,y distance from nearest light sensitive pixel to the origin
    xyf = xyn + det_size - margin           # x,y distance from farthest light sensitive pixel to the origin
    det_lims = {'1': ([-xyf, -xyn], [+xyn, +xyf]),
                '2': ([+xyn, +xyf], [+xyn, +xyf]),
                '3': ([-xyf, -xyn], [-xyf, -xyn]),
                '4': ([+xyf, +xyf], [-xyn, -xyn])
                }

    mfp_xy_lim = 40.0
    # Focal plane bounds (mm)
    fp_bounds = {'efp': (-efp_x_fov_mm/2., +efp_x_fov_mm/2., -efp_y_fov_mm/2., +efp_y_fov_mm/2.),
                 'mfp': (-mfp_xy_lim, +mfp_xy_lim, -mfp_xy_lim, +mfp_xy_lim)
                 }

    mosaic_size = 2. * (det_size - margin) + det_gap
    # Image quality parameters are calculated for three levels of data product
    process_levels = ['raw_zemax', 'proc_zemax', 'proc_detector']
    axes = ['spectral', 'spatial']
    # Zemax PSF image oversampling wrt detector pixels.
    oversampling = 4
