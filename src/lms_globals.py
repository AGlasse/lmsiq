#!/usr/bin/env python
"""
@author Alistair Glasse
Python object to encapsulate LMS optical constants

29/7/19  Created class (Glasse)
"""
import math
from astropy import units as u


class Globals:
    # Constants
    mas_as = 1000.
    deg_rad = 180. / math.pi
    rad_per_mas = 4.85E-9
    sterad_per_mas2 = rad_per_mas * rad_per_mas
    u.plam = u.photon / u.s / u.cm / u.cm / u.angstrom / u.steradian
    u.cm2 = u.cm * u.cm

    # Optical parameters
    elt_area = 1350. * u.m * u.m
    pix_spec_res_el = 2.5       # Pixels per spectral resolution element

    # Simulators
    scopesim, toysim = 'scopesim', 'toysim'

    # Optical configurations and focal planes
    nominal = 'nominal'
    extended = 'extended'
    coord_in = 'efp_x', 'efp_y', 'wavelength'
    coord_out = 'det_x', 'det_y'

    # Zemax data descriptors
    dist_nom_config = ('distortion', nominal, '20240109', 'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
                       coord_in, coord_out)
    dist_ext_config = ('distortion', extended, '20250110', 'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                       coord_in, coord_out)
    iq_nom_config = ('iq', nominal, '2024073000', 'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
                     coord_in, coord_out)
    iq_ext_config = ('iq', extended, '2024061403', 'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                     coord_in, coord_out)

    model_configurations = {'distortion': {nominal: dist_nom_config, extended: dist_ext_config},
                            'iq': {nominal: iq_nom_config, extended: iq_ext_config}
                            }
    lms_config_template = {'opticon': None, 'pri_ang': None, 'ech_ang': None}
    slice_config_template = {'slice_no': None, 'spifu_no': None, 'ech_ord': None, 'w_min': None, 'w_max': None}

    # Transform parameters
    svd_order = 4
    svd_shape = svd_order, svd_order
    matrix_names = ['a', 'b', 'ai', 'bi']
    n_svd_matrices = len(matrix_names)
    svd_cutoff = 1.0e-7                                     # SVD eigenvalues below this value set to zero.
    n_svd_fit_terms = 6
    # Order of polynomial fit to wave = f(pri_ang, ech_ang) and tform_matrix_term = g(pri_ang, ech_ang)
    surface_fit_order = 4  # Matrix order for polynomial surface fit (3 or 4)
    surface_fit_n_coeffs = 10     # Non-zero terms in upper triangular matrix of order 'surface_fit_order'
    wpa_fit_order = {nominal: 6, extended: 3}

    zemax_configuration = None
    n_lms_detectors = 4
    det_pix_size, im_pix_size = None, None
    optical_configurations = [nominal, extended]
    slice_no_ranges = {nominal: range(1, 29), extended: range(12, 15)}
    spifu_no_ranges = {nominal: range(0, 1), extended: range(1, 7)}

    ipc_on_tag, ipc_off_tag = '_ipc_01_3', '_ipc_00_0'      # IPC/diffusion file tags

    slice_id_fmt = "{:s}_{:d}_{:d}_{:02d}_{:d}"        # Define transforms by opticon, ech_ord, slice_no, spifu_no

    # Plate scale at the entrance focal plane.  Defined in LB email 14/11/24 (in ../docs/lb_ps_131124.txt)
    # as efp_as_mm = 350.06 / 1937 = 0.180723
    efp_as_mm = 0.180723 * u.arcsec / u.mm

    # Plate scale at detector
    alpha_pix = 8.7 * u.mas                                   #
    beta_slice = 20.7 * u.mas

    # The field of view in the optical design is quoted in the FDR design report (E-REP-ATC-MET-1003) is then
    alpha_fov = 0.897 * u.arcsec
    beta_fov = beta_slice.to(u.arcsec) * 28
    efp_x_fov_mm = alpha_fov / efp_as_mm   # EFP field of view (mm) (Note ray trace bounds 5.842063, 3.208105)
    efp_y_fov_mm = beta_fov / efp_as_mm

    debug = False
    if debug:
        fmt = "{:>30s} = {:5.3f} x {:5.3f} {:s}"
        print(fmt.format('EFP field of view, alpha, beta', efp_x_fov_mm, efp_y_fov_mm, 'mm'))
        print(fmt.format('', alpha_fov, beta_fov, 'arcseconds'))

    # Diffraction grating parameters
    blaze_angle = 51.23                                     # Echelle blaze angle (deg)
    rule_spacing = 18.2			                            # Echelle rule spacing [um]
    ge_refractive_index = 4.05                              # Refractive index of germanium
    wav_first_order = 2. * rule_spacing * ge_refractive_index
    wav_first_order = 21 * 5.216                            # First order blaze wavelength

    # IFU parameters
    n_lms_slices = 28
    n_lms_spifu_slices = 6

    nom_pix_pitch = 18.0                                    # LMS pixel pitch in microns
    det_gap = 3.0				            # Gap between active regions of detectors in 2 x 2 mosaic (mm)
    pix_margin = [64, 64]		            # Unilluminated margin around outer detector edge (pixels)
    margin = pix_margin[0] * nom_pix_pitch / 1000.          # Convert to mm
    det_format = 2048, 2048
    mosaic_format = 2, 2
    det_size = det_format[0] * nom_pix_pitch / 1000.                 # Detector size in mm
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

    as_built_file = '../output/asbuilt/asbuilt'
