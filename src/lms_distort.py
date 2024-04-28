#!/usr/bin/env python
"""

"""
from os import listdir
from lms_filer import Filer
from lms_util import Util
from lms_dist_plot import Plot
from lms_dist_trace import Trace
from lms_globals import Globals
from lms_detector import Detector

print('lms_distort - Starting')

analysis_type = 'distortion'

coords_out = 'det_x', 'det_y'

nominal = Globals.nominal
nom_coords_in = 'phase', 'fp2_x'
nom_date_stamp = '20240109'  # Old version = 20190627
nom_config = (analysis_type, nominal, nom_date_stamp,
              'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
              nom_coords_in, coords_out)

spifu = Globals.spifu
spifu_coords_in = 'phase', 'fp1_x'
spifu_date_stamp = '20231009'
spifu_config = (analysis_type, spifu, spifu_date_stamp,
                'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                spifu_coords_in, coords_out)

model_configurations = {nominal: nom_config, spifu: spifu_config}

""" SET MODEL CONFIGURATION HERE """
model_config = model_configurations[spifu]
filer = Filer(model_config)

_, optical_path, date_stamp, optical_path_label, coords_in, coords_out = model_config
is_spifu = optical_path == spifu

print("- optical path  = {:s}".format(optical_path))
print("- input coords  = {:s}, {:s}".format(coords_in[0], coords_in[1]))
print("- output coords = {:s}, {:s}".format(coords_out[0], coords_out[1]))

# File locations and names
zem_folder = filer.data_folder

detector = Detector()

mat_names = ['A', 'B', 'AI', 'BI']
focal_planes = {''}

util = Util()
plot = Plot()

n_mats = Globals.n_mats_transform

st_file = open(filer.stats_file, 'w')

run_config = 4, 2
n_terms, poly_order = run_config
st_hdr = "Trace individual"
rt_text_block = ''

suppress_plots = False  # f = Plot first trace
generate_transforms = False  # for all Zemax ray trace files and write to lms_dist_buffer.txt
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
        trace = Trace(zf_file, coords_in, coords_out, silent=True, is_spifu=is_spifu)
        print(trace.__str__())
        debug = not suppress_plots
        trace.create_transforms(n_terms, debug=debug)
        trace.add_wave_bounds(debug=False)
        traces.append(trace)
        offset_data_list.append(trace.offset_data)
        if not suppress_plots:
            trace.plot_focal_planes()
            trace.plot_fit_maps()
            suppress_plots = True  # True = Just plot first file/setting
    print(Filer.trace_file)
    filer.write_pickle(filer.trace_file, traces)
    # stats_data = n_terms, -1, np.array(offset_data_list)
    # util.print_stats(st_file, stats_data)

# Wavelength calibration -
# Create an interpolation object to give,
#   wave = f(order, slice, spifu_slice, prism_angle, ech_angle, det_x, det_y)
suppress_plots = False
derive_wcal = True
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
    plot.wavelength_coverage(traces, optical_path)

# Evaluate the transform performance when mapping test data.  The method is to interpolate the
# coordinates determined using the transforms (stored in the 'trace' objects) for adjacent configurations.
evaluate_transforms = True  # performance statistics, for optimising code parameters.
debug = False
if evaluate_transforms:
    print('lms_distort - Evaluating polynomial fitted primary (AB) transforms')

    traces = filer.read_pickle(filer.trace_file)

    test_efp_x = 0.000
    test_efp_y = 0.000
    test_wave = 4.65
    test_efp_point = test_efp_x, test_efp_y, test_wave

    test_ech_order, test_slice_no, test_spifu_no = 27, 14, -1
    test_ech_angle, test_prism_angle = 0.0, 6.65
    if optical_path == spifu:
        test_ech_order, test_slice_no, test_spifu_no = 27, 14, 3
        test_ech_angle = 0.0, 6.97

    test_config = test_ech_order, test_slice_no, test_spifu_no
    test_det_x, test_det_y = Util.efp_to_dfp(traces, test_efp_point, test_config)

    traces = filer.read_pickle(filer.trace_file)
    for trace in traces:
        ech_angle = trace.parameter['Echelle angle']
        prism_angle = trace.parameter['Prism angle']
        for slice_object in trace.slice_objects:
            config, matrices, offset_corrections, rays, _ = slice_object
            waves, phase, alpha, det_x, det_y, det_x_loc, det_y_loc = rays  # Local slice transformed x, y
            # a_fit, b_fit, _, _ = Util.lookup_transform_fit(config, prism_angle, transform_fits)
            # det_x_fit, det_y_fit = Util.apply_distortion(phase, alpha, a_fit, b_fit)
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

    # offset_x = np.array(det_x_fit) - np.array(det_x)
    # offset_y = np.array(det_y_fit) - np.array(det_y)

    # fmt = "Residuals, n_tran_terms={:d}, n_poly_terms={:d}"
    # fig_title = fmt.format(n_terms, poly_order+1)
    # ax_list = plot.set_plot_area(fig_title, fontsize=22,
    #                              xlabel="Wavelength [micron]",
    #                              ylabel="Mean ray offset at detector [micron]")
    # ax = ax_list[0, 0]
    # x = np.mean(offset_x, axis=(1, 2)) * 1000.0
    # y = np.mean(offset_y, axis=(1, 2)) * 1000.0
    # plot.plot_points(ax, wms, x, colour='blue', ms=6, mk='o')
    # plot.plot_points(ax, wms, y, colour='red', ms=6, mk='+')
    # plot.show()

    # stats_data = n_terms, poly_order, offset_x, offset_y
    # util.print_stats(st_file, stats_data)

st_file.close()
print('lms_distort - Done')
