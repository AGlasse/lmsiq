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

    # The ray trace data extends over an EFP fov which maps every pixel on the detector, including the intra-slice
    # gaps.  We therefore subtract a margin efp_y_margin (mm) to implement the intra-slice gap in simulated images.
    # Inferred from ray trace data (0.171 gives alpha fov = 1.0 as)
    efp_x_fov_rt_mm, efp_y_fov_rt_mm = 5.842063, 3.208105   # Field x, y size (mm) extrapolated from ray trace data
    efp_x_fov_margin = 0.8
    efp_x_fov_mm, efp_y_fov_mm = efp_x_fov_rt_mm - efp_x_fov_margin, efp_y_fov_rt_mm
    alpha_fov_as = 1.0
    efp_as_mm = alpha_fov_as / efp_x_fov_mm                 # Plate scale set to give 1.0 arcsec y field

    beta_fov_as = efp_as_mm * efp_y_fov_mm
    fmt = "Ray trace EFP field of view, alpha, beta = {:7.3f}{:7.3f}"
    print(fmt.format(efp_x_fov_rt_mm * efp_as_mm, efp_y_fov_rt_mm * efp_as_mm))
    fmt = "Unvignetted EFP field, alpha, beta = {:7.3f}{:7.3f}"
    print(fmt.format(alpha_fov_as, beta_fov_as))

    alpha_mas_pix = 8.7                                     #
    beta_mas_pix = 20.7
    # Diffraction grating parameters
    rule_spacing = 18.2			                            # Echelle rule spacing [um]
    ge_refractive_index = 4.05                              # Refractive index of germanium
    wav_first_order = 2. * rule_spacing * ge_refractive_index
    wav_first_order = 21 * 5.216  # First order blaze wavelength

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
    mosaic_size = 2. * (det_size - margin) + det_gap
    # Image quality parameters are calculated for three levels of data product
    process_levels = ['raw_zemax', 'proc_zemax', 'proc_detector']
    axes = ['spectral', 'spatial']
    # Zemax PSF image oversampling wrt detector pixels.
    oversampling = 4
