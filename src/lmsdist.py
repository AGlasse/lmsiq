#!/usr/bin/env python
"""

"""
from os import listdir
import numpy as np
from lms_filer import Filer
from lmsdist_util import Util
from lmsdist_plot import Plot
from lmsdist_trace import Trace
from lms_globals import Globals
from lms_detector import Detector

print('lmsdist, distortion model - Starting')

analysis_type = 'distortion'

coords_in = 'efp_x', 'efp_y', 'wavelength'
coords_out = 'det_x', 'det_y'

nominal = Globals.nominal

nom_date_stamp = '20240109'
nom_config = (analysis_type, nominal, nom_date_stamp,
              'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
              coords_in, coords_out)

spifu = Globals.spifu
spifu_date_stamp = '20231009'
spifu_config = (analysis_type, spifu, spifu_date_stamp,
                'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                coords_in, coords_out)

model_configurations = {nominal: nom_config, spifu: spifu_config}

""" SET MODEL CONFIGURATION HERE """
opticon = spifu

model_config = model_configurations[opticon]
filer = Filer(model_config)

_, opticon, date_stamp, optical_path_label, coords_in, coords_out = model_config
# is_spifu = opticon == spifu

print("- optical path  = {:s}".format(opticon))
print("- input coords  = {:s}, {:s}".format(coords_in[0], coords_in[1]))
print("- output coords = {:s}, {:s}".format(coords_out[0], coords_out[1]))

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

suppress_plots = False                  # f = Plot first trace
generate_transforms = True             # for all Zemax ray trace files and write to lms_dist_buffer.txt
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
    for file_name in file_list:
        print(file_name)
        zf_file = zem_folder + file_name
        trace = Trace(zf_file, coords_in, coords_out, silent=True, opticon=opticon)
        print(trace.__str__())
        trace.create_transforms(debug=False)
        # trace.add_wave_bounds(debug=False)
        traces.append(trace)
        offset_data_list.append(trace.offset_data)
        if not suppress_plots:
            trace.plot_focal_planes()
            trace.plot_fit_maps()
            suppress_plots = True                           # True = Just plot first file/setting
        fits_name = filer.write_fits_transform(trace)
    print(Filer.trace_file)
    filer.write_pickle(filer.trace_file, traces)

# Wavelength calibration -
# Create an interpolation object to give,
#   wave = f(order, slice, spifu_slice, prism_angle, ech_angle, det_x, det_y)
suppress_plots = False
derive_wcal = False
if derive_wcal:
    print()
    print("Generating wavelength calibration file (wavelength <-> echelle angle)")
    traces = Filer.read_pickle(filer.trace_file)
    wcal = {}
    for trace in traces:
        for slice_object in trace.slice_objects:
            config, matrices, offset_corrections, rays, wave_bounds = slice_object
            ech_order, slice_no, spifu_no = config
            a, b, ai, bi = matrices
            waves, phase, alpha, det_x, det_y, det_x_fit, det_y_fit = rays
    plot.series('dispersion', traces, opticon)
    plot.series('coverage', traces, opticon)

# Evaluate the transform performance when mapping test data.  The method is to interpolate the
# coordinates determined using the transforms (stored in the 'trace' objects) for adjacent configurations.
evaluate_transforms = True  # performance statistics, for optimising code parameters.
debug = False
if evaluate_transforms:
    print('lms_distort - Evaluating polynomial fitted primary (AB) transforms')

    fits_transforms = filer.read_fits_transforms()
    # Check out and back for the boresight
    test_efp_x = 3.000          # Along slice position in entrance focal plane (mm)
    test_efp_y = -0.15          # Across slice (mm)
    test_wave = 4.65
    efp_launch = {'efp_x': np.array([test_efp_x]),
                  'efp_y': np.array([test_efp_y]),
                  'efp_w': np.array([test_wave])}
    efp_x_out, efp_y_out = efp_launch['efp_x'][0], efp_launch['efp_y'][0]

    test_slice_no, test_phase = Util.efp_y_to_slice(test_efp_y)
    fmt1 = "EFP coord: wave = {:.3f} um, along slice (efp_x) = {:.3f} mm,"
    fmt2 = "across slice (efp_y) = {:.3f} mm, (slice no. {:d})"
    fmt = fmt1 + fmt2
    print(fmt.format(test_wave, test_efp_x, test_efp_y, test_slice_no))
    transforms = []
    for transform in fits_transforms:
        config = transform['configuration']
        w_min, w_max, slice_no = config['w_min'], config['w_max'], config['slice']
        # Is the transform applicable for the target slice?
        if test_slice_no != slice_no:
            continue
        # Is the ray in the wavelength range covered by the tranform?
        in_wrange = (w_min < test_wave) and (test_wave < w_max)
        if not in_wrange:
            continue
        # Does the transformed ray hit the detector?
        matrices = transform['matrices']
        dfp_points = Util.efp_to_dfp(transform, efp_launch)
        det_x, det_y = dfp_points['det_x'], dfp_points['det_y']
        is_hit, _ = Util.is_det_hit(det_x, det_y)
        if not is_hit:
            continue
        transforms.append(transform)
    fmt = "Ray will be detected in {:d} modelled LMS configurations"
    print(fmt.format(len(transforms)))
    fmt = "{:>12s},{:>12s},{:>20s},{:>20s},{:>20s},{:>12s},{:>12s},{:>12s},{:>12s}"
    print(fmt.format('efp_x_launch', 'efp_y_launch', 'Prism angle', 'Echelle angle', 'Echelle order', 'det_x', 'det_y',
                     'efp_x_back', 'efp_y_back'))
    print(fmt.format('mm', 'mm', 'deg.', 'deg.', '-', 'mm', 'mm', 'mm', 'mm'))
    fmt = "{:12.3f},{:12.3f},{:20.3f},{:20.3f},{:20d},{:12.3f},{:12.3f},{:12.3f},{:12.3f}"
    for transform in transforms:
        config = transform['configuration']
        pri_ang, ech_ang, ech_ord = config['pri_ang'], config['ech_ang'], config['ech_ord']
        matrices = transform['matrices']
        dfp_points = Util.efp_to_dfp(transform, efp_launch)
        det_x, det_y = dfp_points['det_x'], dfp_points['det_y']
        efp_back = Util.dfp_to_efp(transform, dfp_points)
        efp_x_back, efp_y_back = efp_back['efp_x'][0], efp_back['efp_y'][0]
        print(fmt.format(efp_x_out, efp_y_out, pri_ang, ech_ang, ech_ord,
                         det_x[0], det_y[0],
                         efp_x_back, efp_y_back))

st_file.close()
print('lms_distort - Done')
