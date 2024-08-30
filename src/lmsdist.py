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

coord_in = 'efp_x', 'efp_y', 'wavelength'
coord_out = 'det_x', 'det_y'

nominal = Globals.nominal

nom_date_stamp = '20240109'
nom_config = (analysis_type, nominal, nom_date_stamp,
              'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
              coord_in, coord_out)

spifu = Globals.spifu
spifu_date_stamp = '20231009'
spifu_config = (analysis_type, spifu, spifu_date_stamp,
                'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                coord_in, coord_out)

model_configurations = {nominal: nom_config, spifu: spifu_config}

""" SET MODEL CONFIGURATION HERE """
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

suppress_plots = False                  # f = Plot first trace
generate_transforms = False              # for all Zemax ray trace files and write to lms_dist_buffer.txt
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
    debug_first = False
    for file_name in file_list:
        print(file_name)
        zf_file = zem_folder + file_name
        trace = Trace(zf_file, model_config, silent=True)
        print(trace.__str__())
        trace.create_transforms(debug=debug_first)
        debug_first = False
        traces.append(trace)
        offset_data_list.append(trace.offset_data)
        if not suppress_plots:
            trace.plot_fit_maps(plotdiffs=False, subset=True, field=False)
            trace.plot_fit_maps(plotdiffs=True, subset=True, field=False)
            suppress_plots = True                           # True = Just plot first file/setting
        fits_name = filer.write_fits_svd_transform(trace)
        a_rms_list.append(trace.a_rms)

    filer.write_fits_affine_transform(Trace)
    a_rms = np.sqrt(np.mean(np.square(np.array(a_rms_list))))
    print("a_rms = {:10.3f} microns".format(a_rms * 1000.))

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
        for slice in trace.slices:
            config, matrices, rays = slice
            ech_order, slice_no, spifu_no, _, _ = config
            a, b, ai, bi = matrices
            waves, phase, alpha, det_x, det_y, det_x_fit, det_y_fit = rays
    plot.series('dispersion', traces)
    plot.series('coverage', traces)

# Evaluate the transform performance when mapping test data.  The method is to interpolate the
# coordinates determined using the transforms (stored in the 'trace' objects) for adjacent configurations.
evaluate_transforms = True  # performance statistics, for optimising code parameters.
debug = False
if evaluate_transforms:
    print('lms_distort - Evaluating polynomial fitted primary (AB) transforms')

    affines = filer.read_fits_affine_transform(date_stamp)
    svd_transforms = filer.read_fits_svd_transforms()
    # Check out and back for a boresight spectrum.
    n_pts = 40

    test_xywsp = np.zeros((n_pts, 5))
    test_xywsp[:, 0] = [3.000]*n_pts
    test_xywsp[:, 1] = [-0.15]*n_pts
    w_start, w_end = 4.65, 4.85
    dw = (w_end - w_start) / (n_pts - 1)

    test_xywsp[:, 2] = w_start + np.array([w for w in dw * np.arange(0, n_pts)])
    test_slice_nos, test_phases = Util.efp_y_to_slice(test_xywsp[:, 1])
    test_xywsp[:, 3] = test_slice_nos
    test_xywsp[:, 4] = test_phases

    fmt = "{:>7s},{:>7s},{:>7s},{:>8s},{:>9s},{:>9s},{:>12s},{:>9s},{:>12s},{:>9s},{:>9s}{:>8s},{:>8s}"
    print(fmt.format('launch', 'launch', 'launch', 'Prism', 'Echelle', 'Echelle',
                     'mosaic', 'mosaic', 'det', 'det', 'det', 'back', 'back'))
    print(fmt.format('efp_x', 'efp_y', 'efp_w', 'angle', 'angle', 'order',
                     'mfp_x', 'mfp_y', 'no.', 'dfp_x', 'dfp_y', 'efp_x', 'efp_y'))
    print(fmt.format('mm', 'mm', 'micron', 'deg.', 'deg.', '-',
                     'mm', 'mm', '-', 'pix', 'pix', 'mm', 'mm'))
    fmt = "{:7.3f},{:7.3f},{:7.3f},{:8.3f},{:9.3f},{:9d},{:12.3f},{:9.3f},{:12d},{:9.1f},{:9.1f},{:8.3f}{:8.3f}"
    test_transforms = []
    efp_launch_list = []
    efp_launch = {'efp_x': [], 'efp_y': [], 'efp_w': []}

    for i in range(0, n_pts):
        test_wave = test_xywsp[i, 2]
        test_slice_no = test_xywsp[i, 3]

        for transform in svd_transforms:  # Collect all test points served by this transform
            config = transform['configuration']
            slice_no = config['slice']
            if slice_no != test_slice_no:
                continue
            # Is the ray in the wavelength range covered by the tranform?
            w_min, w_max = config['w_min'], config['w_max']
            in_wrange = (w_min < test_wave) and (test_wave < w_max)
            if not in_wrange:
                continue
            # Does the transformed ray hit the detector?
            matrices = transform['matrices']
            [efp_x, efp_y, efp_w] = test_xywsp[i, 0:3]
            efp_points = {'efp_x': [efp_x], 'efp_y': [efp_y], 'efp_w': [efp_w]}
            mfp_points = Util.efp_to_mfp(transform, efp_points)
            dfp_points = Util.mfp_to_dfp(affines, mfp_points)
            mfp_x, mfp_y = mfp_points['mfp_x'], mfp_points['mfp_y']
            is_hit, _ = Util.is_det_hit(mfp_x, mfp_y)
            if not is_hit:
                continue
            efp_back = Util.mfp_to_efp(transform, mfp_points)
            efp_x_back, efp_y_back = efp_back['efp_x'][0], efp_back['efp_y'][0]
            det_nos, dfp_x, dfp_y = dfp_points['det_nos'], dfp_points['dfp_x'], dfp_points['dfp_y']
            pri_ang = config['pri_ang']
            ech_ang = config['ech_ang']
            ech_ord = config['ech_ord']
            print(fmt.format(efp_x, efp_y, efp_w,
                             pri_ang, ech_ang, ech_ord,
                             mfp_x[0], mfp_y[0],
                             det_nos[0], dfp_x[0], dfp_y[0],
                             efp_x_back, efp_y_back))

st_file.close()
print('lms_distort - Done')
