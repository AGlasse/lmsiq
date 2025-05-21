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
from lmsdist_polyfit import PolyFit

print('lmsdist, distortion model - Starting')

analysis_type = 'distortion'
coord_in = 'efp_x', 'efp_y', 'wavelength'
coord_out = 'det_x', 'det_y'

nominal = Globals.nominal
nom_date_stamp = '20240109'
nom_config = (analysis_type, nominal, nom_date_stamp,
              'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
              coord_in, coord_out)

extended = Globals.extended
ext_date_stamp = '20250110'
ext_config = (analysis_type, extended, ext_date_stamp,
                'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
              coord_in, coord_out)

model_configurations = {nominal: nom_config, extended: ext_config}

""" SET MODEL CONFIGURATION HERE """
opticon = extended                       # 'nominal' or 'spifu'

# wpa_fit_order = Globals.wpa_fit_order_dict[opticon]  # Order of 1D polynomial fit, wavelength -> prism angle
model_config = model_configurations[opticon]

_, opticon, date_stamp, optical_path_label, coord_in, coord_out = model_config
filer = Filer(model_config)
polyfit = PolyFit(opticon)

print("- optical path  = {:s}".format(opticon))
print("- input coords  = {:s}, {:s}".format(coord_in[0], coord_in[1]))
print("- output coords = {:s}, {:s}".format(coord_out[0], coord_out[1]))

# File locations and names
zem_folder = filer.data_folder

detector = Detector()

focal_planes = {''}

util = Util()
plot = Plot()

n_mats = Globals.n_svd_matrices
st_file = open(filer.stats_file, 'w')

run_config = 4, 2
n_terms, poly_order = run_config
st_hdr = "Trace individual"
rt_text_block = ''

suppress_plots = True                       # f = Plot first trace
generate_transforms = False
if generate_transforms:
    print()
    print("Generating distortion transforms (and prism angle fit parameters)")
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

    for cfg_id, file_name in enumerate(file_list):
        print(file_name)
        zf_file = zem_folder + file_name
        trace = Trace(zf_file, model_config, cfg_id=cfg_id, silent=True)
        trace.create_transforms(debug=debug_first)
        debug_first = False
        traces.append(trace)
        offset_data_list.append(trace.offset_data)
        if not suppress_plots:
            trace.plot_fit_maps(plotdiffs=False, subset=True, field=True)
            trace.plot_fit_maps(plotdiffs=True, subset=True, field=True)
            trace.plot_focal_planes()
            suppress_plots = False
        fits_name = filer.write_svd_transforms(trace)
        trace.transform_fits_name = fits_name
        a_rms_list.append(trace.a_rms)

    filer.write_affine_transform(Trace)
    a_rms = np.sqrt(np.mean(np.square(np.array(a_rms_list))))
    print("a_rms = {:10.3f} microns".format(a_rms * 1000.))

    print(Filer.trace_file)
    filer.write_pickle(filer.trace_file, traces)

suppress_plots = False
calibrate_wavelength = True
if calibrate_wavelength:
    print()
    print("Plotting wavelength dispersion and coverage for all configurations")
    wcal = {}
    traces = Filer.read_pickle(filer.trace_file)
    plot.series('dispersion', traces)
    plot.series('coverage', traces[0:1])
    plot.series('coverage', traces)

fit_transforms = True
if fit_transforms:
    # Create 2D (prism and echelle angle) polynomial fits to the wavelength and distortion transforms.  The prism angle
    # is provided as a function of wavelength, reflecting the requirement that the target wavelength must be directed
    # through the slit at the pre-disperser output (this is calculated for spatial slice number 13.
    # For the extended mode, the fit directs spectral slice number 3 through the slit.
    _, opticon, _, _, _, _ = filer.model_configuration
    opt_tag = 'nom' if opticon == nominal else 'ext'
    svd_transforms = filer.read_svd_transforms(inc_tags=[opt_tag], exc_tags=['fit_parameters'])

    # Generate and save a polynomial fit for the function prism_angle(wavelength).
    traces = Filer.read_pickle(filer.trace_file)
    wave_boresights, prism_angles = [], []   # Wavelength which passes through Int Foc Plane origin.
    for trace in traces:
        wave_bs, pri_ang = trace.get_ifp_boresight(opticon)
        wave_boresights.append(wave_bs)
        prism_angles.append(pri_ang)
    wpa_fit  = polyfit.create_pa_wave_fit(opticon, wave_boresights, prism_angles)
    wxo_fit, wxo_header, term_fits = polyfit.create_polynomial_surface_fits(opticon, svd_transforms)
    filer.write_fit_parameters(wpa_fit, wxo_fit, wxo_header, term_fits)
    plot.wave_v_prism_angle(wpa_fit, polyfit.wpa_model, wave_boresights, prism_angles)

# Evaluate the transform performance by comparing the coordinates of the Zemax ray trace with the projected
# coordinates using 1) the specific at the Zemax location and 2) the model fit transforms (generated for the prism
# and echelle angles)
evaluate_transforms = True
if evaluate_transforms:
    traces = Filer.read_pickle(filer.trace_file)            # Use the ray trace data
    _, opticon, date_stamp, _, _, _ = filer.model_configuration
    inc_tags = ["efp_mfp_{:s}".format(opticon[0:3])]
    svd_transforms = filer.read_svd_transforms(inc_tags=inc_tags, exc_tags=['fit_parameters'])
    for trace in traces:
        inc_tags = [trace.transform_fits_name]
        trace_transforms = filer.read_svd_transforms(inc_tags=inc_tags)
        wpa_fit, wxo_fit, term_fits = filer.read_fit_parameters(opticon)
        mfp_projections = {}
        slice_nos = [28, 13, 1] if opticon == nominal else [13]
        spifu_nos = [0] if opticon == nominal else [1, 3, 6]
        for slice_no in slice_nos:
            mfp_projections[slice_no] = {}
            for spifu_no in spifu_nos:
                mfp_projections[slice_no][spifu_no] = {}
                slice_transform = Util.filter_transform_list(trace_transforms,
                                                             slice_no=slice_no, spifu_no=spifu_no)[0]
                mfp_projection = PolyFit.make_mfp_projection(slice_no, spifu_no, trace, slice_transform,
                                                             wxo_fit, term_fits)
                mfp_projections[slice_no][spifu_no] = mfp_projection
        do_plot = False
        if do_plot:
            plot.mfp_projections(mfp_projections, trace)

test_transform_fit = True
if test_transform_fit:
    print_header = True
    test_waves = np.linspace(2.7, 5.4, 28, endpoint=True)
    if opticon == Globals.extended:
        test_waves = np.linspace(4.4, 4.6, 3, endpoint=True)
    for test_wave in np.linspace(2.7, 5.4, 28, endpoint=True):  # Test wavelength does not match an SVD transform
        _, opticon, date_stamp, _, _, _ = filer.model_configuration
        wpa_fit, wxo_fit, term_fits = filer.read_fit_parameters(opticon)
        PolyFit.wave_to_config(test_wave, wpa_fit, wxo_fit, debug=False, select='min_ech_ang', print_header=print_header)
        print_header = False
    print_header = True
    for test_wave in np.linspace(2.7, 2.9, 6, endpoint=True):  # Test wavelength does not match an SVD transform
        _, opticon, date_stamp, _, _, _ = filer.model_configuration
        wpa_fit, wxo_fit, term_fits = filer.read_fit_parameters(opticon)
        PolyFit.wave_to_config(test_wave, wpa_fit, wxo_fit, debug=False, select='min_ech_ang', print_header=print_header)
        print_header = False

print()
print('lms_distort - Done')
