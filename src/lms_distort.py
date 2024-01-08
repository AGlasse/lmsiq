#!/usr/bin/env python
"""

"""
from os import listdir
from lms_filer import Filer
from lms_dist_util import Util
from lms_dist_plot import Plot
from lms_dist_trace import Trace
from lms_globals import Globals
from lms_detector import Detector
import numpy as np

print('lms_distort - Starting')

nominal = 'nominal'
spifu = 'spifu'
optics = {nominal: ('Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)', ('phase', 'fp2_x'), ('det_x', 'det_y')),
          spifu: ('Extended spectral coverage (fov = 1.0 x 0.054 arcsec)', ('phase', 'fp1_x'), ('det_x', 'det_y'))}
optical_configuration = nominal

config_summary, coord_in, coord_out = optics[optical_configuration]
is_spifu = optical_configuration == spifu
filer = Filer(optical_configuration)

print("- optical path  = {:s}".format(optical_configuration))
print("- input coords  = {:s}, {:s}".format(coord_in[0], coord_in[1]))
print("- output coords = {:s}, {:s}".format(coord_out[0], coord_out[1]))

# File locations and names
zem_folder = "../data/distortion/{:s}".format(optical_configuration)
zem_folder = Filer.get_folder(zem_folder)

detector = Detector()

mat_names = ['A', 'B', 'AI', 'BI']
focal_planes = {''}

util = Util()
plot = Plot()

n_mats = Globals.n_mats_transform

st_file = open(Filer.stats_file, 'w')

run_config = 4, 2       # Nominal 4, 2
n_terms, poly_order = run_config
st_hdr = "Trace individual"
rt_text_block = ''

suppress_plots = False          # f = Plot first trace
generate_transforms = True     # for all Zemax ray trace files and write to lms_dist_buffer.txt
if generate_transforms:
    print()
    print("Generating distortion transforms")
    fmt = "- reading Zemax ray trace data from folder {:s}"
    print(fmt.format(zem_folder))

    file_list = listdir(zem_folder)
    n_traces = len(file_list)
    offset_data_list = []
    traces = []
    for file_name in file_list:
        print(file_name)
        zf_file = zem_folder + file_name
        trace = Trace(zf_file, coord_in, coord_out, silent=True, is_spifu=is_spifu)
        print(trace.__str__())
        trace.create_transforms(n_terms, debug=False)
        trace.add_wave_bounds(debug=False)
        traces.append(trace)
        offset_data_list.append(trace.offset_data)
        if not suppress_plots:
            # trace.plot_dispersion(poly_order=0)         # Level shift
            # trace.plot_dispersion(poly_order=1)
            trace.plot_scatter(slice_list=[14.])
            trace.plot_focal_planes()
            trace.plot_fit_maps()
            suppress_plots = False       # True = Just plot first file/setting
    Filer.write_pickle(Filer.trace_file, traces)
    stats_data = n_terms, -1, np.array(offset_data_list)
    util.print_stats(st_file, stats_data)

# Wavelength calibration -
# Create an interpolation object to give,
#   wave = f(order, slice, spifu_slice, prism_angle, ech_angle, det_x, det_y)
suppress_plots = False
derive_wcal = True
if derive_wcal:
    print()
    print("Generating wavelength calibration file (wavelength <-> echelle angle)")
    traces = Filer.read_pickle(Filer.trace_file)
    wcal = {}
    for trace in traces:
        for tf in trace.tf_list:
            config, matrices, rays, wave_bounds = tf
            ech_order, slice_no, spifu_no = config
            a, b, ai, bi = matrices
            waves, phase, alpha, det_x, det_y, det_x_fit, det_y_fit = rays
    plot.wavelength_coverage(traces)

# Generate polynomial fits to transform matrix coefficients. (per slice and per echelle order)
generate_polynomials = True    # which interpolate transforms to any echelle angle / order
if generate_polynomials:
    fmt = "Generating polynomial fit to wavelength v transform term for each slice and echelle order"
    print(fmt)
    fmt = "- polynomial order = {:d}"
    print(fmt.format(poly_order))
    traces = Filer.read_pickle(Filer.trace_file)
    transform_tuple = util.get_transform_fits(traces, poly_order)
    Filer.write_pickle(Filer.tf_fit_file, transform_tuple)

# Evaluate the transform performance when mapping the trace data using the polynomial fitted
# transforms calculated in tf_fit_file.
evaluate_transforms = True     # performance statistics, for optimising code parameters.
debug = True
if evaluate_transforms:
    print('lms_distort - Evaluating polynomial fitted primary (AB) transforms')
    det_x_fit, det_y_fit = None, None
    transform_tuple = Filer.read_pickle(Filer.tf_fit_file)
    transform_configs, transform_fits = transform_tuple

    traces = Filer.read_pickle(Filer.trace_file)
    for trace in traces:
        ech_angle = trace.parameter['Echelle angle']
        prism_angle = trace.parameter['Prism angle']
        for tf in trace.tf_list:
            config, matrices, rays, _ = tf
            waves, phase, alpha, det_x, det_y, det_x_loc, det_y_loc = rays     # Local slice transformed x, y
            a_fit, b_fit, _, _ = Util.lookup_transform_fit(config, prism_angle, transform_fits)
            det_x_fit, det_y_fit = Util.apply_distortion(phase, alpha, a_fit, b_fit)
            if debug:
                nrows, ncols = 7, 4
                fig_title = "{:s} ea = {:4.2f}".format(trace.parameter['name'], ech_angle)
                ax_list = plot.set_plot_area(fig_title)
                ax = ax_list[0, 0]
                args = {'clip_on': True, 'fillstyle': 'none', 'mew': 1., 'linestyle': 'None',
                        'marker': '+', 'ms': 6., 'color': 'black', 'label': 'Zemax'}
                p1 = ax.plot(det_x, det_y, **args)
                args['color'], args['marker'] = 'blue', 'x'
                p2 = ax.plot(det_x_loc, det_y_loc, **args)
                args['color'], args['marker'] = 'red', 's'
                p3 = ax.plot(det_x_fit, det_y_fit, **args)
                ax.legend(handles=[p1[0], p2[0], p3[0]])
                Plot.show()

    offset_x = np.array(det_x_fit) - np.array(det_x)
    offset_y = np.array(det_y_fit) - np.array(det_y)

    fmt = "Residuals, n_tran_terms={:d}, n_poly_terms={:d}"
    fig_title = fmt.format(n_terms, poly_order+1)
    ax_list = plot.set_plot_area(fig_title, fontsize=22,
                                 xlabel="Wavelength [micron]",
                                 ylabel="Mean ray offset at detector [micron]")
    ax = ax_list[0, 0]
    x = np.mean(offset_x, axis=(1, 2)) * 1000.0
    y = np.mean(offset_y, axis=(1, 2)) * 1000.0
    plot.plot_points(ax, wms, x, colour='blue', ms=6, mk='o')
    plot.plot_points(ax, wms, y, colour='red', ms=6, mk='+')
    plot.show()

    stats_data = n_terms, poly_order, offset_x, offset_y
    util.print_stats(st_file, stats_data)

st_file.close()
print('lms_distort - Done')
