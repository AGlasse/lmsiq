#!/usr/bin/env python
"""

"""
import glob
from os.path import join
from lms_util import Util
from lms_plot import Plot
from lms_trace import Trace
from lms_globals import Globals
from lms_wcal import Wcal
import numpy as np

print('lms_distort - Starting')

# Set up run control flags.
generate_transforms = True         # ..for all Zemax ray trace files and write to lms_dist_buffer.txt
derive_wcal = True                  # Find echelle_order(wavelength) and echell_angle(wavelength)
generate_polynomials = True         # ..which interpolate transforms to any echelle angle / order
evaluate_transforms = True          # ..performance statistics, for optimising code parameters.

mat_names = ['A', 'B', 'AI', 'BI']

util = Util()
plot = Plot()

n_mats = Globals.n_mats_transform
n_slices = Globals.n_slices
n_rays_slice = 400

run_configs = [[4, 3]]  #[[2, 1], [2, 2], [3, 2], [4, 2], [4, 3], [5, 2]]
st_file = open(Globals.stats_file, 'w')
util.print_stats(st_file, None)     # Print header

for run in run_configs:
    n_terms = run[0]
    poly_order = run[1]
    st_hdr = "Trace individual"

    if generate_transforms:
        print('- Generating primary LMS mapping transform file from Zemax data')
        tf_file = util.openw_transform_file(Globals.transform_file, n_terms)
        suppress = True                                                             # f = Plot first trace
        file_list = glob.glob(join(Globals.zem_folder, 'LMS_*'), recursive=True)    # Zemax trace files
        n_traces = len(file_list)
        ox = np.zeros((n_traces, n_slices, n_rays_slice))
        oy = np.zeros((n_traces, n_slices, n_rays_slice))

        for f in range(0, n_traces):
            zf_file = file_list[f]
            trace = Trace(zf_file, silent=True)
            tf_abs = trace.to_transforms(n_terms)
            text = trace.tfs_to_text(tf_abs)
            tf_file.write(text)

            offset_x, offset_y = trace.get_transform_offsets(tf_abs)
            ox[f, :, :] = offset_x
            oy[f, :, :] = offset_y
            if not suppress:
                trace.plot_fit_maps(offset_x[f], offset_y[f], plotdiffs=False)
                trace.plot(suppress=True)
                trace.plot_slice_map(10)
                suppress = True
        tf_file.close()
        stats_data = n_terms, -1, ox, oy
        util.print_stats(st_file, stats_data)

    configs_list = []
    if derive_wcal:       # Plot wavelength coverage for all configurations
        tf_list = util.read_transform_file(Globals.transform_file)
        configs = util.extract_configs(tf_list)
        wcal = Wcal()
        wcal.write_poly(configs)
        # Check wavelength settings..
        wcal.read_poly()
        plot.wavelengths_v_ech_ang(configs)

    # Generate polynomial fits to transform matrix coefficients. (per slice and per echelle order)
    if generate_polynomials:
        print('lms_distort - Generating new polynomial fit transforms, order = {:d}'.format(poly_order))
        tf_list = util.read_transform_file(Globals.transform_file)
        util.write_polyfits_file(tf_list, n_terms, poly_order)

    # Evaluate the transform performance when mapping the trace data.
    if evaluate_transforms:
        print('lms_distort - Evaluating primary (AB) transforms')
        poly, ech_bounds = util.read_polyfits_file(Globals.poly_file)

        # Project the trace data using the polynomial fit transforms.
        file_list = glob.glob(join(Globals.zem_folder, 'LMS_*'), recursive=True)
        n_traces = len(file_list)
        ox = np.zeros((n_traces, n_slices, n_rays_slice))
        oy = np.zeros((n_traces, n_slices, n_rays_slice))
        eas = np.zeros(n_traces)
        eos = np.zeros(n_traces)
        pas = np.zeros(n_traces)
        wms = np.zeros(n_traces)

        suppress = True
        for f in range(0, n_traces):
            file = file_list[f]
            trace = Trace(file, silent=True)
            ea = trace.echelle_angle
            eo = trace.echelle_order
            pa = trace.prism_angle
            wm = trace.mean_wavelength
            eas[f] = ea
            wms[f] = wm
            eos[f] = eo
            pas[f] = pa
            if not suppress:
                nrows = 7
                ncols = 4
                fig_title = "{:s} ea = {:4.2f}".format(trace.name, ea)
                ax_list = plot.set_plot_area(fig_title, nrows=nrows, ncols=ncols, xlim=[-60.0, 60.0])
            for s in range(0, n_slices):
                sno = s + 1
                phase = trace.get('Phase', slice=sno)
                alpha = trace.get('FP2_X', slice=sno)
                fp6_x = trace.get('FP6_X', slice=sno)
                fp6_y = trace.get('FP6_Y', slice=sno)
                a, b, ai, bi = util.get_polyfit_transform(poly, ech_bounds, eo, s, ea)
    #            label = "Slice {:6d}, Echelle angle {:8.2f}".format(s, ea)
    #            util.print_poly_transform((a, b, ai, bi), label=label)
                det_x, det_y = util.apply_distortion(phase, alpha, a, b)
                if not suppress:
                    row = s % nrows
                    col = int(s / nrows)
                    ax = ax_list[row, col]
                    plot.plot_points(ax, det_x, det_y, ms=1.0, colour='red')
                    plot.plot_points(ax, fp6_x, fp6_y, ms=1.0, mk='x')
                offset_x = det_x - fp6_x
                offset_y = det_y - fp6_y
                ox[f, s, :] = offset_x
                oy[f, s, :] = offset_y
            if not suppress:
                plot.show()
            suppress = True

        plot_residuals = True
        if plot_residuals:
            fmt = "Residuals, n_tran_terms={:d}, n_poly_terms={:d}"
            fig_title = fmt.format(n_terms, poly_order+1)
            ax_list = plot.set_plot_area(fig_title, fontsize=22,
                                         xlabel="Wavelength [micron]",
                                         ylabel="Mean ray offset at detector [micron]")
            ax = ax_list[0,0]
            x = np.mean(ox, axis=(1,2)) * 1000.0
            y = np.mean(oy, axis=(1,2)) * 1000.0
            plot.plot_points(ax, wms, x, colour='blue', ms=6, mk='o')
            plot.plot_points(ax, wms, y, colour='red', ms=6, mk='+')
            plot.show()

        stats_data = n_terms, poly_order, ox, oy
        util.print_stats(st_file, stats_data)

st_file.close()
print('lms_distort - Done')
