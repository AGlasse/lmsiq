#!/usr/bin/env python
""" Program to generate transform which map coordinates between the LMS entrance focal plane
EFP_w, wavelength,
EFP_x, along slice position (mm)
EFP_y, across slice position,
and detector focal plane
DET_NO, detector number (slices 1 to 14 fall on detectors 3 and 4, 15 to 28 on dets 1 and 2, and
                         1,2 image the short wavelengths)
DFP_x, along row position
DFP_y, along row position
"""
import numpy as np
from os import listdir
from lms_filer import Filer
from lmsdist_util import Util
from lmsdist_plot import Plot
from lmsdist_trace import Trace
from lms_globals import Globals
from lms_detector import Detector
from lms_transform import Transform
from lmsdist_polyfit import PolyFit

print('lmsdist, distortion model - Starting')

analysis_type = 'distortion'            # Used for file handling (types = 'iq', 'distortion' or 'sky'

""" SET MODEL CONFIGURATION HERE """
opticon = Globals.extended                     # 'nominal' or 'extended'

filer = Filer()
filer.set_configuration(analysis_type, opticon)
polyfit = PolyFit(opticon)

print("- optical path  = {:s}".format(opticon))
print("- input coords  = {:s}, {:s}".format(Globals.coord_in[0], Globals.coord_in[1]))
print("- output coords = {:s}, {:s}".format(Globals.coord_out[0], Globals.coord_out[1]))

# File locations and names
zem_folder = filer.data_folder

detector = Detector()

focal_planes = {''}

util = Util()
plot = Plot()

n_mats = Globals.n_svd_matrices
filer.set_configuration(analysis_type, opticon)
st_file = open(filer.stats_file, 'w')

run_config = 4, 2
n_terms, poly_order = run_config
st_hdr = "Trace individual"
rt_text_block = ''

generate_transforms = False
do_first_plot = True
if generate_transforms:
    print()
    print("Generating distortion transforms (and prism angle fit parameters)")
    fmt = "- reading Zemax ray trace data from folder {:s}"
    print("Plot first configuration = {:s}".format(str(do_first_plot)))
    print(fmt.format(zem_folder))

    # Select *.csv files
    file_list = listdir(zem_folder)
    file_list = [f for f in file_list if '.csv' in f]

    n_traces = len(file_list)
    offset_data_list = []
    traces = []
    a_rms_list = []
    debug_first = True          # Print debugging information for first trace only.

    for cfg_id, file_name in enumerate(file_list):
        print(file_name)
        zf_file = zem_folder + file_name
        trace = Trace(cfg_id=cfg_id, silent=True)
        trace.load(zf_file, filer.model_configuration, do_plot=False)
        trace.find_wavelength_bounds()
        debug_first = False
        offset_data_list.append(trace.offset_data)
        if do_first_plot:
            trace.plot_focal_planes()
            trace.plot_fit_maps(plotdiffs=False, subset=True, field=True)
            # trace.plot_fit_maps(plotdiffs=True, subset=True, field=True)
            do_first_plot = False

        fits_name = filer.write_svd_transforms(trace, do_plot=True)
        trace.transform_fits_name = fits_name
        traces.append(trace)
        a_rms_list.append(trace.a_rms)

    filer.write_affine_transform(Trace)
    a_rms = np.sqrt(np.mean(np.square(np.array(a_rms_list))))
    print("a_rms = {:10.3f} microns".format(a_rms * 1000.))
    print(Filer.trace_file)
    Filer.write_dill(Filer.trace_file, traces)

calibrate_wavelength = True
if calibrate_wavelength:
    first_plot = True
    print()
    print("Plotting wavelength dispersion and coverage for all configurations")
    wcal = {}
    traces = Filer.read_dill(filer.trace_file)
    model_config = filer.model_configuration
    # plot.series('nm_det', traces, model_config)
    if first_plot:
        plot.series('dispersion', traces, model_config)
        plot.series('coverage', traces[0:1], model_config)
        plot.series('coverage', traces, model_config)
        first_plot = False

fit_transforms = True
is_first = True
if fit_transforms:
    # Create 2D (prism and echelle angle) polynomial fits to the wavelength and distortion transforms.  The prism angle
    # is provided as a function of wavelength, reflecting the requirement that the target wavelength must be directed
    # through the slit at the pre-disperser output (this is calculated for spatial slice number 13).
    # For the extended mode, the fit directs spectral slice number 3 through the slit.
    # _, opticon, _, _, _, _ = filer.model_configuration

    # Generate and save a polynomial fit for the function pri_ang = prism_angle(wavelength).  This function
    # is intended to return the prism angle which will map a wavelength to the centre of the mosaic (mfp_x = 0.)
    # when the echelle angle is equal to zero.  We don't have complete ray trace data for ech_ang = 0, so we
    # interpolate across available trace ech_ang values on a per order basis.  We also don't have sufficient ray trace
    # data for the extended mode, so we use the nominal mode fit, which should be identical since the MSA comes after
    # the pre-disperser.
    opt_tag = 'ext'
    if opticon == Globals.nominal:
        opt_tag = 'nom'
        all_boresights = []                         # All boresights, including non-zero echelle angles
        boresight_waves, prism_angles = [], []      # Wavelength which passes through Int Foc Plane origin.
        traces = Filer.read_dill(filer.trace_file)
        for trace in traces:
            boresight = trace.get_ifp_boresight(opticon)
            all_boresights.append(boresight)
        all_boresights = np.array(all_boresights)        # Row content is wavelength, pri_ang, ech_ang, ech_ord
        ech_orders = all_boresights[:, 3]
        unique_ech_orders = np.unique(ech_orders)
        for ech_order in unique_ech_orders:
            idx = ech_orders == ech_order
            order_boresight_waves = all_boresights[idx, 0]
            order_prism_angles = all_boresights[idx, 1]
            order_ech_angles = all_boresights[idx, 2]
            ea_wave_fit = polyfit.create_polynomial_fit(order_ech_angles, order_boresight_waves, poly_order=3)
            ea_wave_coeff = ea_wave_fit['wpa_opt']
            boresight_wave = polyfit.poly_model(0., *ea_wave_coeff)     # Wavelength for ech_ang = 0.
            boresight_waves.append(boresight_wave)
            wave_pa_fit = polyfit.create_polynomial_fit(order_boresight_waves, order_prism_angles, poly_order=3)
            wave_pa_coeff = wave_pa_fit['wpa_opt']
            boresight_pa = polyfit.poly_model(boresight_wave, *wave_pa_coeff)
            prism_angles.append(boresight_pa)
        poly_order = Globals.wpa_fit_order[opticon]
        wpa_fit  = polyfit.create_polynomial_fit(boresight_waves, prism_angles, poly_order=poly_order)
        if is_first:
            plot.wave_v_prism_angle(wpa_fit, polyfit.poly_model, boresight_waves, prism_angles)
            is_first = False
    else:
        filer.set_configuration('distortion', Globals.nominal)
        wpa_fit, _, _ = filer.read_fit_parameters(Globals.nominal)
        print('Reading in ''nominal'' mode prism calibration for use in extended mode. ')

    # Create transform term fits and write to file.
    filer.set_configuration(analysis_type, opticon)
    svd_transforms = filer.read_svd_transforms(inc_tags=[opt_tag], exc_tags=['fit_parameters'])
    wxo_fit, wxo_header, svd_fit = polyfit.create_polynomial_surface_fits(opticon, svd_transforms, plot_wxo=False)
    filer.write_fit_parameters(wpa_fit, wxo_fit, wxo_header, svd_fit)



# Evaluate the transform performance by comparing the coordinates of the Zemax ray trace with the projected
# coordinates.  using 1) the specific transform for the trace at the Zemax location, 2) the model fit transforms
# (generated for the prism and echelle angles).
evaluate_transforms = True
if evaluate_transforms:
    traces = Filer.read_dill(filer.trace_file)            # Use the ray trace data
    _, opticon, date_stamp, _, _, _ = filer.model_configuration
    affines = filer.read_fits_affine_transform(date_stamp)
    spifu_no = 0
    inc_tags = ["efp_mfp_{:s}".format(opticon[0:3])]
    wpa_fit, wxo_fit, svd_fit = filer.read_fit_parameters(opticon)

    fmt = '{:45s},{:8s},{:8s},{:8s},{:10s},{:10s},{:10s},{:1s},{:10s},{:10s},{:1s},{:10s},{:10s},{:1s},'
    print(fmt.format('Trace', 'slice_no', 'spifu_no', 'ech_ord', 'wave_ref',
                     'Zemax', 'Zemax', '|', 'Zemax', 'Zemax', '|', 'PolyFit', 'PolyFit', '', '', '|'))
    print(fmt.format('file name', '-', '-', '-', 'efp_w0',
                     'pri_ang', 'ech_ang', '|', 'mfp_x0', 'mfp_y0', '|', 'mfp_x0', 'mfp_y0', '|'))
    fmt = '{:45s},{:8d},{:8d},{:8d},{:10.3f},{:10.3f},{:10.3f},{:1s},{:10.3f},{:10.3f},{:1s},{:10.3f},{:10.3f},{:1s}'
    for trace in traces:
        lms_config, wave_bs = None, None
        inc_tags = [trace.transform_fits_name]
        trace_transforms = filer.read_svd_transforms(inc_tags=inc_tags, exc_tags=['fit_parameters'])

        slice_nos = trace.unique_slices
        spifu_nos = trace.unique_spifu_slices
        ech_ords = trace.unique_ech_ords

        mfp_plot = {'slice_no': [], 'spifu_no': [], 'ech_ord': [], 'ray': [], 'sli': [], 'fit': []}
        for slice_no in slice_nos:
            for spifu_no in spifu_nos:
                for ech_ord in ech_ords:
                    slice_filter = {'slice_no':slice_no, 'spifu_no':spifu_no, 'ech_ord':ech_ord}
                    # Start by finding the slice transform for this configuration
                    slice_transform_zem = None
                    for slice_transform_zem in trace_transforms:
                        is_match = slice_transform_zem.is_match(slice_filter)
                        if is_match:
                            break

                    efp_w = trace.get_series('wavelength', slice_filter)
                    if len(efp_w) < 1:      # spifu_no and ech_ord are not independent.
                        continue
                    efp_x = trace.get_series('efp_x', slice_filter)
                    efp_y = trace.get_series('efp_y', slice_filter)
                    efp_points = {'efp_x': efp_x, 'efp_y': efp_y, 'efp_w': efp_w}
                    mfp_pts_sli_tform, _ = util.efp_to_mfp(slice_transform_zem, efp_points)
                    det_x = trace.get_series('det_x', slice_filter)
                    det_y = trace.get_series('det_y', slice_filter)
                    mfp_pts_ray = {'mfp_x': det_x, 'mfp_y': det_y}
                    mfp_plot['ray'].append(mfp_pts_ray)
                    mfp_plot['sli'].append(mfp_pts_sli_tform)
                    mfp_plot['slice_no'].append(slice_no)
                    mfp_plot['spifu_no'].append(spifu_no)
                    mfp_plot['ech_ord'].append(ech_ord)

                    slice_config_zem = slice_transform_zem.slice_configuration
                    fit_matrix = svd_fit[slice_no][spifu_no]
                    lms_config = trace.lms_config
                    pri_ang = lms_config['pri_ang']
                    ech_ang = lms_config['ech_ang']
                    n_terms = Globals.svd_order
                    matrices = {}
                    for mat_name in Globals.matrix_names:
                        matrix = np.zeros((n_terms, n_terms))
                        matrices[mat_name] = matrix
                        for row in range(0, n_terms):
                            for col in range(0, n_terms):
                                fit_terms = fit_matrix[mat_name][row, col]
                                term = PolyFit.surface_model((pri_ang, ech_ang), *fit_terms)
                                matrix[row, col] = term

                    slice_transform_fit = Transform(matrices=matrices, slice_config=slice_config_zem, lms_config=lms_config)
                    mfp_pts_fit_tform, _ = util.efp_to_mfp(slice_transform_fit, efp_points)
                    mfp_plot['fit'].append(mfp_pts_fit_tform)

                    det_x_fit = mfp_pts_fit_tform['mfp_x']
                    det_y_fit = mfp_pts_fit_tform['mfp_y']
                    wave_bs = np.mean(efp_w)
                    text = fmt.format(trace.transform_fits_name, slice_no, spifu_no, ech_ord, wave_bs,
                                      pri_ang, ech_ang, '|', det_x[0], det_y[0], '|', det_x_fit[0], det_y_fit[0], '|'
                                      )
                    print(text)

        do_plot = True
        if do_plot:
            theta_p = r'$\phi_{pri}$' + "={:6.3f}, ".format(lms_config['pri_ang'])
            theta_e = r'$\psi_{ech}$' + "={:6.3f}, ".format(lms_config['ech_ang'])
            wave_text = r'$\lambda_{bs}$' + "={:6.3f}, ".format(wave_bs) + r'$\mu$m'

            title = theta_p + theta_e + wave_text
            mfp_decorations = [('ray trace', 'red', 'o', 10.),
                               ('slice transform', 'green', 'x', 6.),
                               ('fit transform', 'blue', '+', 6.)]
            Plot.plot_mfp_points(mfp_plot, mfp_decorations, title=title, ref_xy=False, grid=False)
            Plot.plot_mfp_points(mfp_plot, mfp_decorations, title=title, ref_xy=True, grid=True)

test_transform_fit = True
if test_transform_fit:
    test_waves = np.linspace(2.7, 5.4, 28, endpoint=True)
    if opticon == Globals.extended:
        test_waves = np.linspace(4.4, 4.6, 3, endpoint=True)
    print_header = True
    for test_wave in np.linspace(2.7, 5.4, 28, endpoint=True):  # Test wavelength does not match an SVD transform
        _, opticon, date_stamp, _, _, _ = filer.model_configuration
        wpa_fit, wxo_fit, svd_fit = filer.read_fit_parameters(opticon)
        PolyFit.wave_to_config(test_wave, opticon, wpa_fit, wxo_fit, debug=True, select='min_ech_ang', print_header=print_header)
        print_header = False
    print_header = True
    for test_wave in np.linspace(2.7, 2.9, 6, endpoint=True):  # Test wavelength does not match an SVD transform
        _, opticon, date_stamp, _, _, _ = filer.model_configuration
        wpa_fit, wxo_fit, svd_fit = filer.read_fit_parameters(opticon)
        PolyFit.wave_to_config(test_wave, opticon, wpa_fit, wxo_fit, debug=False, select='min_ech_ang', print_header=print_header)
        print_header = False

print()
print('lms_distort - Done')
