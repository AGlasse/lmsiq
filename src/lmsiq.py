import os
import numpy as np
from lms_globals import Globals
from lms_filer import Filer
from lms_wcal import Wcal
from lms_ipc import Ipc
from lms_detector import Detector
from lmsiq_analyse import Analyse
from lmsiq_fitsio import FitsIo
from lmsiq_plot import Plot
from lmsiq_phase import Phase
from lmsiq_cuber import Cuber
from lmsiq_summariser import Summariser
from lmsiq_image_manager import ImageManager
from lms_dist_util import Util

print('LMS Repeatability (lmsiq.py) - started')
print()

analysis_type = 'iq'

# Define the optical path, either nominal or with the spectral IFU inserted in the beam.
nominal = Globals.nominal
spifu = Globals.spifu
optics = {nominal: ('Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)', ('phase', 'fp2_x'), ('det_x', 'det_y')),
          spifu: ('Extended spectral coverage (fov = 1.0 x 0.054 arcsec)', ('phase', 'fp1_x'), ('det_x', 'det_y'))}

# Locations (slices numbers and IFU) where Zemax images are provided.
# The target PSF is centred on the first slice in the list.  The number of adjacent
# slices is next, then optional 'ifu' if images at the IFU image slicer are present
slice_locs_20230607 = ['14', '2']
slice_locs_20230630 = ['14', '4']
# Configuation tuple - dataset, optical_configuration, n_orders, n_runs, im_locs, folder_label, label

# nominal_identifier_old = {'optical_configuration': nominal,
#                       'iq_date_stamp': '20230630',
#                       'iq_folder_leader': 'psf_model',
#                       'dist_date_stamp': '20240109',
#                       'zemax_format': 'old_zemax',
#                       'specifu_config_file': None,
#                       'ifu_setup': slice_locs_20230630,
#                       }
nominal_identifier = {'optical_configuration': nominal,
                      'iq_date_stamp': '20240209',
                      'iq_folder_leader': 'psf_IFU2det_',
                      'dist_date_stamp': '20240109',
                      'zemax_format': 'new_zemax',
                      'specifu_config_file': None,
                      'cube_slice_bounds': (13, 1),
                      }
# spifu_identifier_old = {'optical_configuration': spifu,
#                     'iq_date_stamp': '20240124',
#                     'iq_folder_leader': 'psf_model',
#                     'dist_date_stamp': '20240109',
#                     'zemax_format': 'new_zemax',
#                     'specifu_config_file': 'SpecIFU_config.csv',
#                     'ifu_setup': (11, 100, ['1', '1', 'ifu']),
#                     }
spifu_identifier = {'optical_configuration': spifu,
                    'iq_date_stamp': '20240209',
                    'iq_folder_leader': 'psf_model',
                    'dist_date_stamp': '20240109',
                    'zemax_format': 'new_zemax',
                    'specifu_config_file': 'SpecIFU_config.csv',
                    'cube_slice_bounds': (13, 1),
                    }

data_identifier = nominal_identifier
optical_path = data_identifier['optical_configuration']
date_stamp = data_identifier['iq_date_stamp']
is_spifu = optical_path == Globals.spifu

fmt = "Analysing dataset for {:s} optical path, dated {:s}"
print(fmt.format(optical_path, date_stamp))

# Initialise static classes
globals = Globals()
analyse = Analyse()
plot = Plot()
detector = Detector()
util = Util()
fitsio = FitsIo(optical_path)

model_configuration = analysis_type, optical_path, date_stamp
iq_filer = Filer(model_configuration)

image_manager = ImageManager()
image_manager.make_dictionary(data_identifier, iq_filer)

summariser = Summariser()

run_test = True  # Generate test case with 50 % IPC and an image with one bright pixel.
print("Running test case = {:s}".format(str(run_test)))
ipc = Ipc()

inter_pixels = [True, False]                        # True = include diffusion kernel convolution

# Analyse Zemax data
process_phase_data = False
if process_phase_data:
    print()
    print("Processing centroid shift impact for dataset {:s}".format(date_stamp))
    phase = Phase()                                 # Module for phase (wavelength) shift analysis
    mc_start, mc_end = 0, 3
    print("Setting mc_start={:d}, mc_end={:d}".format(mc_start, mc_end))

    mc_bounds = mc_start, mc_end
    process_control = mc_bounds, inter_pixels
    phase.process(data_identifier, process_control, iq_filer, image_manager, plot_level=1)

build_cubes = True
if build_cubes:
    print('Reconstructing cubes and analysing slice profile data')
    cuber = Cuber()
    rt_date_stamp = '20231009' if is_spifu else '20240109'
    rt_model_configuration = 'distortion', optical_path, rt_date_stamp
    rt_filer = Filer(rt_model_configuration)
    cuber.build(data_identifier, inter_pixels, image_manager, iq_filer)

process_cubes = False
if process_cubes:
    print('Processing cubes to extract slice by slice profiles and reconstructed image Strehls (and more)')
    cuber = Cuber()
    cuber.process(data_identifier, inter_pixels, iq_filer)

#poly_file = '../output/lms_dist_poly_old.txt'           # Distortion polynomial
#poly, ech_bounds = Util.read_polyfits_file(poly_file)   # Use distortion map for finding dispersion

# Read in dataset centroids for plotting.
stats_list = []
plot_id = 0, 'blue', 'solid', 'D'

process_level = 'raw_zemax'
profile_data = None

plot_profiles = True
if plot_profiles:
    data_identifier = data_identifiers[dataset_to_analyse]
    optical_path, date_stamp, n_wavelengths, n_mcruns, slice_locs, folder_name, config_label = data_identifier
    n_mcruns_tag = "{:04d}".format(n_mcruns)
    # Get boresight slice only
    tgt_slice_no, n_slices, slice_idents = Util.parse_slice_locations(slice_locs, boresight=True)
    profile_data = None

    # Dictionary of profiles to plot. Includes y axis label and a boolean True = Data has errors
    profile_labels = {'fwhm_spec_lin_mc': ('FWHM spectral (linear)', True),
                      'fwhm_spec_lin_per': ('FWHM spectral (perfect, linear)', False),
                      'fwhm_spec_lin_des': ('FWHM spectral (design, linear)', False),
                      'fwhm_spec_gau_mc': ('FWHM spectral (gaussian)', True),
                      'srp_mc_lin': ('<SRP> M-C (linear)', True),
                      'srp_mc_gau': ('<SRP> M-C (gaussian)', True)
                      }
    for profile in profile_labels:
        for slice_ident in slice_idents:
            slice_no, slice_subfolder, slice_label = slice_ident
            if slice_no != tgt_slice_no:
                continue
            process_levels = Globals.process_levels
            for process_level in Globals.process_levels:
                profile_data_list = []
                for inter_pixel in inter_pixels:
                    Ipc.set_inter_pixel(inter_pixel)
                    profile_dict, profiles = Summariser.read_summary(process_level, slice_subfolder, iq_filer)
                    profile_data = plot_id, data_identifier, profile_dict, profiles, Ipc.tag
                    profile_data_list.append(profile_data)
                png_folder = iq_filer.iq_png_folder + '/profiles'
                png_folder = iq_filer.get_folder(png_folder)
                png_file = process_level + '_' + profile
                png_path = png_folder + png_file
                ylabel, plot_errors = profile_labels[profile]
                srp_req = 'srp' in profile
                plot.profile(profile, profile_data_list,
                             ylabel=ylabel, ls='solid',
                             srp_req=srp_req, plot_errors=plot_errors, png_path=png_path)

print('LMS Repeatability (lmsiq.py) - done')
