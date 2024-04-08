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
from lmsiq_test import Test

print('LMS Repeatability (lmsiq.py) - started')
print()

analysis_type = 'iq'

# Define the optical path, either nominal or with the spectral IFU inserted in the beam.
nominal = Globals.nominal
spifu = Globals.spifu
optics = {nominal: ('Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)', ('phase', 'fp2_x'), ('det_x', 'det_y')),
          spifu: ('Extended spectral coverage (fov = 1.0 x 0.054 arcsec)', ('phase', 'fp1_x'), ('det_x', 'det_y'))}

nominal_identifier = {'optical_configuration': nominal,
                      'iq_date_stamp': '20240324',
                      'iq_folder_leader': 'psf_',
                      'mc_bounds': None,                    # Set to None to use all M-C data
                      'dist_date_stamp': '20240305',
                      'zemax_format': 'new_zemax',
                      'specifu_config_file': None,
                      'field_bounds': (7, 8, 9),
                      'cube_slice_bounds': (23, 4),         # {'123': 13, '456': 5, '789': 23}
                      }
spifu_identifier = {'optical_configuration': spifu,
                    'iq_date_stamp': '20240324',
                    'iq_folder_leader': 'psf_',
                    'mc_bounds': None,                   # Set to speed program when debugging.
                    'dist_date_stamp': '20240109',
                    'zemax_format': 'new_zemax',
                    'specifu_config_file': None,
                    'field_bounds': (1, 2, 3),
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
ipc = Ipc()

run_test = False  # Generate test case with 50 % IPC and an image with one bright pixel.
print("\nRunning test case = {:s}".format(str(run_test)))
if run_test:
    test = Test()
    test.run(iq_filer)

inter_pixels = [True, False]                        # True = include diffusion kernel convolution
mc_bounds = image_manager.model_dict['mc_bounds']
if data_identifier['mc_bounds'] is not None:
    mc_bounds = data_identifier['mc_bounds']
process_control = mc_bounds, inter_pixels
print("\nSetting mc_start={:d}, mc_end={:d}".format(mc_bounds[0], mc_bounds[1]))

# Analyse Zemax data
process_phase_data = False
if process_phase_data:
    print()
    print("Processing centroid shift impact for dataset {:s}".format(date_stamp))
    phase = Phase()                                 # Module for phase (wavelength) shift analysis
    phase.process(data_identifier, process_control, iq_filer, image_manager,
                  config_nos=[0, 20],       # =[0, 20] for speed, =None to analyse all configs
                  plot_level=2)

build_cubes = True
if build_cubes:
    print()
    print('\nReconstructing cubes and analysing slice profile data')
    print('-----------------------------------------------------')
    cuber = Cuber()
    rt_date_stamp = '20231009' if is_spifu else '20240109'
    rt_model_configuration = 'distortion', optical_path, rt_date_stamp, None, None, None
    rt_filer = Filer(rt_model_configuration)
    cuber.build(data_identifier, process_control, image_manager, iq_filer)

plot_series = True
if plot_series:
    print('Plot cube series data (wavelength/field) and calculate key statistics')
    cuber = Cuber()
    cube_pkl_folder = iq_filer.get_folder(iq_filer.cube_folder + 'pkl')

    uni_par = image_manager.unique_parameters
    field_nos = uni_par['field_nos']
    cube_series, cubes = {}, []
    is_first_field = True
    for field_no in field_nos:
        field_tag = "field_{:d}_".format(field_no)
        field_series_path = cube_pkl_folder + 'cube_series_' + field_tag
        field_series, field_cubes = iq_filer.read_pickle(field_series_path)
        if is_first_field:
            for key in field_series:
                cube_series[key] = []
                is_first_field = False
        for key in field_series:
            cube_series[key] += field_series[key]
        cubes += field_cubes
    cube_series_plot = Cuber.remove_configs(cube_series, [21])
    cuber.plot_series(optical_path, cube_series_plot, iq_filer)

print('LMS Repeatability (lmsiq.py) - done')
