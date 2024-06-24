import os
import numpy as np
import time
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

# data_tags = {'nom': 'nominal', 'ext': 'extended',
#              'tor': 'toroidal M12', 'sph': 'spherical M12',
#              'dfp': 'detector focal plane fields 10-12',
#              'efp': 'entrance focal plane field 1-9'}
# Data descriptor: optical_path, mc_bounds, dist_date_stamp, field_tgt_slice, slice_radius, label
fts_dfp = {10: 1, 11: 13, 12: 28}
fts_efp = {1: 13, 2: 5, 3: 13, 4: 5, 5: 5, 6: 5, 7: 24, 8: 24, 9: 24}
data_descriptor = {'2024032400': (nominal, 'all', '20240305', fts_efp, 0, "full L- and M-bands"),
                   '20240324': (spifu, None, '20240109', {1: 13, 2: 13, 3: 13}, 1, "extended coverage, toroidal M12"),
                   '2024043000': (nominal, 'all', '20240305', fts_dfp, 0,
                                  "torM12, defocus .0,.05,.1 mm, wave 2.7 um"),
                   '2024050700': (nominal, 'all', '20240305', fts_dfp, 0,
                                  "torM12, DFP fields 10-12, defocus .0,.05,.1 mm, wave 5.0 um"),
                   '2024060700': (nominal, 'all', '20240305', fts_dfp, 0, 'spherical M12, defocus   0 um'),
                   '2024060705': (nominal, 'all', '20240305', fts_dfp, 0, 'spherical M12, defocus  50 um'),
                   '2024060710': (nominal, 'all', '20240305', fts_dfp, 0, 'spherical M12, defocus 100 um'),
                   '2024060720': (nominal, 'all', '20240305', fts_dfp, 0, 'spherical M12, defocus 200 um'),

                   '2024060800': (nominal, 'all', '20240305', fts_dfp, 0,
                                  "spherical M12, EFP fields 1-9, full LM-bands"),
                   '2024061800': (nominal, 'all', '20240305', fts_dfp, 0,
                                  "METIS_sphericalM12_M19manufacture_errors - WFE adjusted"),
                   '2024061801': (nominal, 'all', '20240305', fts_dfp, 0,
                                  "METIS_sphericalM12_M19manufacture_errors - WFE all")
                   }

nominal_identifier = {'optical_path': nominal,
                      'iq_date_stamp': '2024060700',
                      'iq_folder_leader': 'psf_',
                      'mc_bounds': 'all',                    # Set to None, [mlo, mhi] or 'all'
                      'dist_date_stamp': '20240305',
                      'specifu_config_file': None,
                      'field_tgt_slice': fts_dfp,
                      'slice_radius': 0,
                      }
spifu_identifier = {'optical_path': spifu,
                    'iq_date_stamp': '20240324',
                    'iq_folder_leader': 'psf_',
                    'mc_bounds': None,                      # Set to speed program when debugging.
                    'dist_date_stamp': '20240109',
                    'specifu_config_file': 'all',
                    'field_tgt_slice': {1: 13, 2: 13, 3: 13},
                    'slice_radius': 1,
                    }

date_stamp = '2024060700'
optical_path, mc_bounds, dist_date_stamp, field_tgt_slice, slice_radius, data_label = data_descriptor[date_stamp]


data_identifier = nominal_identifier

# optical_path = data_identifier['optical_path']
# date_stamp = data_identifier['iq_date_stamp']
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

inter_pixels = [True]                        # True = include diffusion kernel convolution
mc_bounds = image_manager.model_dict['mc_bounds']
req_mc_bounds = data_identifier['mc_bounds']
if req_mc_bounds is not None:
    if req_mc_bounds != 'all':
        mc_bounds = mc_bounds if req_mc_bounds == 'all' else req_mc_bounds
        print("\nSetting mc_start={:d}, mc_end={:d}".format(mc_bounds[0], mc_bounds[1]))
process_control = mc_bounds, inter_pixels

# Analyse Zemax data
process_phase_data = False
if process_phase_data:
    # Calculate the impact of sub-pixel shifts on photometry and line centroiding.
    print("\nProcessing centroid shift impact for dataset {:s}".format(date_stamp))
    phase = Phase()                                 # Module for phase (wavelength) shift analysis
    phase.process(data_identifier, process_control, iq_filer, image_manager,
                  config_nos=[0], field_nos=[2], plot_level=2)

build_cubes = True
if build_cubes:
    print()
    print('\nReconstructing cubes and analysing slice profile data')
    print('-----------------------------------------------------')
    cuber = Cuber()
    rt_date_stamp = '20231009' if is_spifu else '20240109'
    rt_model_configuration = 'distortion', optical_path, rt_date_stamp, None, None, None
    rt_filer = Filer(rt_model_configuration)
    cuber.build(data_identifier, process_control, image_manager, iq_filer, debug=False)

plot_cubes = True
if plot_cubes:
    title = '\nPlot cube series data (wavelength/field) and calculate key statistics'
    print(title)
    print('-'*len(title))
    cuber = Cuber()
    cube_packages = cuber.read_pkl_cubes(iq_filer)
    if optical_path == 'nominal':           # Filter out weird nominal configuration...!
        cube_packages = Cuber.remove_configs(cube_packages, [21])
    # cuber.write_csv(optical_path, cube_packages, iq_filer)
    cuber.plot(optical_path, cube_packages, iq_filer, is_defocus=True)

print('LMS Repeatability (lmsiq.py) - done')
