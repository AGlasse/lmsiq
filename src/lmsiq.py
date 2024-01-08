import numpy as np
from lms_globals import Globals
from lms_filer import Filer
from lms_wcal import Wcal
from lms_ipc import Ipc
from lms_ipg import Ipg
from lms_detector import Detector
from lmsiq_analyse import Analyse
from lmsiq_fitsio import FitsIo
from lmsiq_plot import Plot
from lmsiq_phase import Phase
from lmsiq_cuber import Cuber
from lmsiq_summariser import Summariser
from lms_dist_util import Util

print('LMS Repeatability (lmsiq.py) - started')
print()

# Define the optical path, either nominal or with the spectral IFU inserted in the beam.
nominal = 'nominal'
spifu = 'spifu'
optics = {nominal: ('Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)', ('phase', 'fp2_x'), ('det_x', 'det_y')),
          spifu: ('Extended spectral coverage (fov = 1.0 x 0.054 arcsec)', ('phase', 'fp1_x'), ('det_x', 'det_y'))}

# List of locations (slices numbers and IFU) where Zemax images are provided.
# The target PSF is centred on the first slice in the list.  The number of adjacent
# slices is next, then optional 'ifu' if images at the IFU image slicer are present
slice_locs_20230607 = ['14', '5', 'ifu']
slice_locs_20230630 = ['14', '9', 'ifu']
# Configuation tuple - dataset, optical_configuration, n_orders, n_runs, im_locs, folder_label, label
datasets = {0: (nominal, '20230207', 11, 100, ['1', '1'], '_with_toroidalM12_', 'toroidal M12, slice  1'),
            1: (nominal, '20230607', 3, 5, slice_locs_20230607, '_with_toroidalM12_', 'toroidal M12'),
            2: (nominal, '20230630', 2, 10, slice_locs_20230630, '_with_toroidalM12_', 'toroidal M12')   # 21, 100
           }
dataset_to_analyse = 2
dataset = datasets[dataset_to_analyse]
optical_path = dataset[0]
date_stamp = dataset[1]
fmt = "Analysing dataset for {:s} optical path, dated {:s}"
print(fmt.format(optical_path, date_stamp))

# Initialise static classes
globals = Globals()
analyse = Analyse()
plot = Plot()
detector = Detector()
util = Util()
fitsio = FitsIo(optical_path)
Filer(optical_path)
Filer.setup_folders(dataset)
summariser = Summariser()


# File locations and names
output_folder = "../output/distortion/{:s}".format(optical_path)
output_folder = Filer.get_folder(output_folder)

raytrace_file = output_folder + optical_path + '_raytrace.txt'

copy_from_zip = False        # Copy Zemax organised files into 'data' folder
if copy_from_zip:
    print("Copying and slice sorting Zemax data from '../zip/' to '../data/")
    FitsIo.copy_from_zip(dataset)

# Get Zemax parameters from first wavelength data to set up global oversampling factors
Globals.det_pix_size = Detector.det_pix_size
im_pix_size, zemax_configuration = FitsIo.setup_zemax_configuration(dataset)
Globals.im_pix_size = im_pix_size
Globals.zemax_configuration = zemax_configuration

run_test = False  # Generate test case with 50 % IPC and an image with one bright pixel.
print("Running test case = {:s}".format(str(run_test)))
ipc = Ipc()
Ipc.test()
inter_pixels = [True, False]                    # True = include diffusion kernel convolution

# Analyse Zemax data
process_phase_data = False
if process_phase_data:
    print()
    print("Processing centroid shift impact for dataset {:s}".format(date_stamp))
    phase = Phase()                             # Module for phase (wavelength) shift analysis
    phase.process(dataset, inter_pixels)

build_cubes = True
if build_cubes:
    print('Reconstructing cubes and analysing slice profile data')
    cuber = Cuber()
    cuber.build(dataset, inter_pixels, raytrace_file)

process_cubes = True
if process_cubes:
    print('Processing cubes to extract slice by slice profiles and reconstructed image Strehls (and more)')
    cuber = Cuber()
    cuber.process(dataset, inter_pixels)

poly_file = '../output/lms_dist_poly_old.txt'           # Distortion polynomial
poly, ech_bounds = Util.read_polyfits_file(poly_file)   # Use distortion map for finding dispersion

# Read in dataset centroids for plotting.
stats_list = []
plot_id = 0, 'blue', 'solid', 'D'

process_level = 'raw_zemax'
profile_data = None

plot_profiles = True
if plot_profiles:
    dataset = datasets[dataset_to_analyse]
    optical_path, date_stamp, n_wavelengths, n_mcruns, slice_locs, folder_name, config_label = dataset
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
                    profile_dict, profiles = Summariser.read_summary(process_level, slice_subfolder)
                    profile_data = plot_id, dataset, profile_dict, profiles, Ipc.tag
                    profile_data_list.append(profile_data)

                png_path = Filer.png_path + '/profiles/' + process_level + '_' + profile
                ylabel, plot_errors = profile_labels[profile]
                srp_req = 'srp' in profile
                plot.profile(profile, profile_data_list,
                             ylabel=ylabel, ls='solid',
                             srp_req=srp_req, plot_errors=plot_errors, png_path=png_path)

print('LMS Repeatability (lmsiq.py) - done')
