#!/usr/bin/env python
"""
"""
import numpy as np
from astropy import units as u
from os import listdir
from lms_filer import Filer
from lmsdist_util import Util
from lmsdist_plot import Plot
from lmsdist_trace import Trace
from lms_globals import Globals
from lms_detector import Detector

print('lmsdist, distortion model - Starting')

analysis_type = 'distortion'
coord_in = 'efp_x', 'efp_y', 'wavelength'
coord_out = 'det_x', 'det_y'

nominal = Globals.nominal
nom_date_stamp = '20240109'
nom_config = (analysis_type, nominal, nom_date_stamp,
              'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
              coord_in, coord_out)

spifu = Globals.extended
spifu_date_stamp = '20250110'
spifu_config = (analysis_type, spifu, spifu_date_stamp,
                'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                coord_in, coord_out)

model_configurations = {nominal: nom_config, spifu: spifu_config}

""" SET MODEL CONFIGURATION HERE 'nominal' or 'spifu' """
opticon = nominal

model_config = model_configurations[opticon]
filer = Filer(model_config)

_, opticon, date_stamp, optical_path_label, coord_in, coord_out = model_config

print("- optical path  = {:s}".format(opticon))
print("- input coords  = {:s}, {:s}".format(coord_in[0], coord_in[1]))
print("- output coords = {:s}, {:s}".format(coord_out[0], coord_out[1]))

# File locations and names
zem_folder = filer.data_folder

detector = Detector()

mat_names = ['A', 'B', 'AI', 'BI']
focal_planes = {''}

util = Util()
plot = Plot()

n_mats = Globals.transform_config['n_mats']
st_file = open(filer.stats_file, 'w')

run_config = 4, 2
n_terms, poly_order = run_config
st_hdr = "Trace individual"
rt_text_block = ''

suppress_plots = True                      # f = Plot first trace
generate_transforms = False                 # for all Zemax ray trace files and write to lms_dist_buffer.txt
if generate_transforms:
    print()
    print("Generating distortion transforms")
    fmt = "- reading Zemax ray trace data from folder {:s}"
    print(fmt.format(zem_folder))

    # Select *.csv files
    file_list = listdir(zem_folder)
    file_list = [f for f in file_list if '.csv' in f]

    n_traces = len(file_list)
    offset_data_list = []
    traces = []
    a_rms_list = []
    debug_first = True
    for file_name in file_list:
        print(file_name)
        zf_file = zem_folder + file_name
        trace = Trace(zf_file, model_config, silent=True)
        trace.create_transforms(debug=debug_first)
        debug_first = False
        traces.append(trace)
        offset_data_list.append(trace.offset_data)
        if not suppress_plots:
            trace.plot_fit_maps(plotdiffs=False, subset=True, field=True)
            trace.plot_fit_maps(plotdiffs=True, subset=True, field=True)
            trace.plot_focal_planes()
            suppress_plots = False
        fits_name = filer.write_fits_svd_transform(trace)
        a_rms_list.append(trace.a_rms)

    filer.write_fits_affine_transform(Trace)
    a_rms = np.sqrt(np.mean(np.square(np.array(a_rms_list))))
    print("a_rms = {:10.3f} microns".format(a_rms * 1000.))

    print(Filer.trace_file)
    filer.write_pickle(filer.trace_file, traces)

suppress_plots = False
plot_wcal = False
if plot_wcal:
    print()
    print("Plotting wavelength dispersion and coverage for all configurations")
    traces = Filer.read_pickle(filer.trace_file)
    wcal = {}
    plot.series('dispersion', traces)
    plot.series('coverage', traces[0:1])
    plot.series('coverage', traces)

fit_transforms = True
if fit_transforms:
    # Wavelength calibration -
    # Create an interpolation object to map a wavelength to echelle and prism angle settings.
    # This is done by using the transforms to derive fits of the form
    #   theta_prism = f(wave),   for theta_echelle = 0. deg
    #   theta_echelle = g(wave, order)
    # Read in transforms and plot term variations with prism and echelle angles.
    test_wave = 4700 * u.nm
    affines = filer.read_fits_affine_transform(date_stamp)
    svd_transforms = filer.read_fits_svd_transforms()
    slice_fits = []
    for slice_no in range(1, 29):
        slice_fit = Util.get_term_values(svd_transforms, slice_no)
        slice_fit = Util.add_wxo_fit(slice_fit)
        plot.wxo_fit(slice_fit)
        slice_fit = Util.add_term_fit(slice_fit)
        f_residuals = plot.transform_fit(slice_fit, do_plots=True)
        f_residuals = plot.transform_fit(slice_fit, do_plots=True, plot_residuals=True)
        slice_fits.append(slice_fit)
    filer.write_fits_fit_parameters(slice_fits)

# Evaluate the transform performance when mapping test data.  The method is to interpolate the
# coordinates determined using the transforms (stored in the 'trace' objects) for adjacent configurations.
evaluate_transforms = False  # performance statistics, for optimising code parameters.
if evaluate_transforms:
    Util.test_out_and_back(filer, opticon, date_stamp)

print()
print('lms_distort - Done')
