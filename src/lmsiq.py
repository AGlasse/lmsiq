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

# Mapping from field number to target slice
fts_dfp = {10: 1, 11: 13, 12: 28}
fts_efp = {1: 13, 2: 13, 3: 13, 4: 5, 5: 5, 6: 5, 7: 24, 8: 24, 9: 24}
fts_spifu = {}
for f in range(0, 18):
    fts_spifu[f+1] = 12 + f % 3

# Data descriptor: optical_path, mc_bounds, dist_date_stamp, field_to_tgt_slice_map, slice_radius, label
data_dictionary = {'2024032400': (nominal, 'all', '20240109', fts_efp, 4,
                                  "toroidal M12, EFP fields 1-9, full LM-bands"),
                   '2024032401': (spifu, 'all', '20240109', {1: 13, 2: 13, 3: 13}, 1,
                                  "extended coverage, toroidal M12"),
                   '2024043000': (nominal, 'all', '20240305', fts_dfp, 0,
                                  "torM12, defocus .0,.05,.1 mm, wave 2.7 um"),
                   '2024050700': (nominal, 'all', '20240305', fts_dfp, 0,
                                  "torM12, DFP fields 10-12, defocus .0,.05,.1 mm, wave 4.57 um"),
                   '2024050701': (nominal, 'all', '20240305', fts_dfp, 0,
                                  "torM12, DFP fields 10-12, defocus .0,.05,.1 mm, wave 5.00 um"),
                   '2024050702': (nominal, 'all', '20240305', fts_dfp, 0,
                                  "torM12, DFP fields 10-12, defocus .0,.05,.1 mm, wave 5.24 um"),

                   '2024060700': (nominal, 'all', '20240305', fts_dfp, 0, 'spherical M12, defocus   0 um'),
                   '2024060705': (nominal, 'all', '20240305', fts_dfp, 0, 'spherical M12, defocus  50 um'),
                   '2024060710': (nominal, 'all', '20240305', fts_dfp, 0, 'spherical M12, defocus 100 um'),
                   '2024060720': (nominal, 'all', '20240305', fts_dfp, 0, 'spherical M12, defocus 200 um'),

                   '2024073000': (nominal, 'all', '20240109', fts_efp, 4,
                                  "spherical M12, EFP fields 1-9, full LM-bands"),
                   '2024061403': (spifu, 'all', '20231009', fts_spifu, 0,
                                  "METIS_M19manufacture_errors (updated)"),
                   }

iq_date_stamp = '2024061403'

optical_path, mc_bounds, dist_date_stamp, field_tgt_slice, slice_radius, data_label = data_dictionary[iq_date_stamp]
data_identifier = {'optical_path': optical_path,
                   'iq_date_stamp': iq_date_stamp,
                   'mc_bounds': mc_bounds,                    # Set to None, [mlo, mhi] or 'all'
                   'dist_date_stamp': dist_date_stamp,
                   'field_tgt_slice': field_tgt_slice,
                   'slice_radius': slice_radius,
                   'data_label': data_label
                   }
is_spifu = optical_path == Globals.spifu

fmt = "Analysing dataset for {:s} optical path, dated {:s}"
print(fmt.format(optical_path, iq_date_stamp))

# Initialise static classes
globals = Globals()
analyse = Analyse()
plot = Plot()
detector = Detector()
util = Util()
fitsio = FitsIo(optical_path)

model_configuration = analysis_type, optical_path, iq_date_stamp, None, None, None
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

build_cubes = False
if build_cubes:
    print()
    print('\nReconstructing cubes and analysing slice profile data')
    print('-----------------------------------------------------')
    cuber = Cuber()
    # dist_date_stamp = '20231009' if is_spifu else '20240109'
    dist_model_configuration = 'distortion', optical_path, dist_date_stamp, None, None, None
    dist_filer = Filer(dist_model_configuration)
    cuber.build(data_identifier, process_control, image_manager, iq_filer, dist_filer, debug=False)

plot_cubes = True
if plot_cubes:
    title = '\nPlot cube series data (wavelength/field) and calculate key statistics'
    print(title)
    print('-'*len(title))
    cuber = Cuber()
    cube_packages = cuber.read_pkl_cubes(iq_filer)
    if optical_path == 'nominal':           # Filter out weird nominal configuration...!
        cube_packages = Cuber.remove_configs(cube_packages, [21])
    # cuber.write_csv(image_manager, cube_packages, iq_filer)
    cuber.plot(optical_path, cube_packages, iq_filer, is_defocus=False)

print('LMS Repeatability (lmsiq.py) - done')
