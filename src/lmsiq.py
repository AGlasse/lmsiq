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
from lms_util import Util

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
                      'iq_date_stamp': '20240305',
                      'iq_folder_leader': 'psf_',
                      'dist_date_stamp': '20240305',
                      'zemax_format': 'new_zemax',
                      'specifu_config_file': None,
                      'cube_slice_bounds': (13, 1),
                      }
spifu_identifier = {'optical_configuration': spifu,
                    'iq_date_stamp': '20240209',
                    'iq_folder_leader': 'psf_',
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

model_configuration = analysis_type, optical_path, date_stamp, None, None, None
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
    mc_bounds = 0, 3
    mc_bounds = image_manager.model_dict['mc_bounds']
    print("Setting mc_start={:d}, mc_end={:d}".format(mc_bounds[0], mc_bounds[1]))
    process_control = mc_bounds, inter_pixels
    phase.process(data_identifier, process_control, iq_filer, image_manager, plot_level=1)

cube_pkl_folder = iq_filer.get_folder(iq_filer.cube_folder + 'pkl')
cube_series_path = cube_pkl_folder + 'cube_series'
build_cubes = False
if build_cubes:
    print('Reconstructing cubes and analysing slice profile data')
    cuber = Cuber()
    rt_date_stamp = '20231009' if is_spifu else '20240109'
    rt_model_configuration = 'distortion', optical_path, rt_date_stamp, None, None, None
    rt_filer = Filer(rt_model_configuration)
    cube_series = cuber.build(data_identifier, inter_pixels, image_manager, iq_filer)
    iq_filer.write_pickle(cube_series_path, cube_series)

process_cubes = True
if process_cubes:
    print('Plot cube series data (wavelength/field) and calculate key statistics')
    if cube_series_path is None:
        print('!! Cube series data not found, run with build_cubes=True !!')
    else:
        cuber = Cuber()
        cube_series = iq_filer.read_pickle(cube_series_path)
        cuber.plot_series(cube_series, iq_filer)

print('LMS Repeatability (lmsiq.py) - done')
