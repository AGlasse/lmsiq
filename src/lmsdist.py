#!/usr/bin/env python
"""
"""
import math
import time
import numpy as np
import scipy.signal

from os import listdir
from lms_filer import Filer
from lmsdist_util import Util
from lmsdist_plot import Plot
from lmsdist_trace import Trace
from lms_globals import Globals
from lms_detector import Detector
from lms_toysim import ToySim

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

suppress_plots = False                      # f = Plot first trace
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
        print(trace.__str__())
        trace.create_transforms(debug=debug_first)
        debug_first = False
        traces.append(trace)
        offset_data_list.append(trace.offset_data)
        if not suppress_plots:
            trace.plot_fit_maps(plotdiffs=False, subset=True, field=True)
            trace.plot_fit_maps(plotdiffs=True, subset=True, field=True)
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
plot_wcal = False
if plot_wcal:
    print()
    print("Plotting wavelength dispersion and coverage for all configurations")
    traces = Filer.read_pickle(filer.trace_file)
    wcal = {}
    plot.series('dispersion', traces)
    plot.series('coverage', traces)

# Evaluate the transform performance when mapping test data.  The method is to interpolate the
# coordinates determined using the transforms (stored in the 'trace' objects) for adjacent configurations.
evaluate_transforms = True  # performance statistics, for optimising code parameters.
debug = False
if evaluate_transforms:

    if debug:
        Util.test_out_and_back(filer, date_stamp)

    # Simulate sky emission data etc...
    # - project EFP slices to detector and write slices to cartoon image
    # - convolve with PSF per slice.
    toysim = ToySim()
    t_start = time.perf_counter()

    # Read header and data shape (only) in from the template.
    data_exts = [1, 2, 3, 4]
    hdr_primary, data_list = filer.read_fits('../data/sim_template.fits', data_exts=data_exts)
    det_shape = data_list[0].shape
    _, n_det_cols = det_shape

    # Define one or more (point-like) pinhole masks which will spatially filter the extended source.
    cfo_pnh = {'name': 'wcupnh', 'efp_x': 0., 'efp_y': 0.}

    pinhole = cfo_pnh
    pinhole['slice_range'] = -4, 4
    slice_no_cen, _ = Util.efp_y_to_slice(pinhole['efp_y'])  # Pinhole is centred on this slice
    pinhole['slice_no_cen'] = slice_no_cen

    use_pinholes = False

    # Add slice extended illumination
    # Define EFP coordinates and target wavelength for spectrum
    w_cen = 4.65
    w_cen_nm = int(w_cen * 1000.)
    dit = 60.  # Integration time in seconds.
    ext_source = 'sky'     # Extended source spectrum, may be 'sky', or 'bbTTTT' for a black body
    pnh_tag = pinhole['name'] if use_pinholes else ''
    sim_config = "lms_{:s}_{:s}_{:s}_{:d}".format(opticon[0:3], ext_source, pnh_tag, w_cen_nm)
    print("Running simulation {:s}".format(sim_config))

    # Find EFP bounds
    efp_xmax = Globals.efp_x_fov_mm
    xh = efp_xmax / 2.
    wrange = [w_cen-.1, w_cen+.1]

    # Load high resolution atmospheric or WCU bb spectrum for target wavelength range (units ph/s/m2/as2/um)
    w_ext, f_ext_in, f_units_ext_in = None, None, None
    tau_metis = 0.1 * Detector.qe
    if ext_source[0:2] == 'bb':
        tau_bb = 0.1 * 1.0                      # Assume 10 % integrating sphere and an attenuation setting
        tbb = float(ext_source[2:])
        w_ext, f_bb, f_units_ext_in = toysim.build_bb_emission(wave_range=wrange, tbb=tbb)
        f_ext_in = tau_metis * tau_bb * f_bb
    if ext_source == 'sky':
        w_ext, f_sky, f_units_ext_in = toysim.load_sky_emission(wave_range=wrange)
        f_ext_in = tau_metis * f_sky
    if ext_source == 'laser4650':
        w_ext = np.arange(4.649, 4.651, 0.001)
        f_las = np.full(w_ext.shape, 10000)
        f_units_ext_in = 'phot/s/m2/um/arcsec2'

    atel = math.pi * (39. / 2)**2               # ELT collecting area
    alpha_pix = Globals.alpha_mas_pix / 1000.   # Along slice pixel scale
    beta_slice = Globals.beta_mas_pix / 1000.   # Slice width
    delta_w = w_cen / 100000                    # Spectral resolution
    pix_delta_w = 2.5                           # Pixels per spectral resolution element
    f_ext = f_ext_in * Detector.qe * atel * alpha_pix * beta_slice * delta_w / pix_delta_w       # el/pixel/second
    f_units = 'el/s/pix'
    print('- converted to el/sec/pix')

    w_pnh, f_pnh_in, f_pnh_units = toysim.build_bb_emission(wave_range=wrange, tbb=1000.)
    tau_wcu_attenuator = 1E-4
    f_pnh = f_pnh_in * tau_metis * tau_wcu_attenuator * atel * alpha_pix * beta_slice * delta_w / pix_delta_w  # el/pixel/second

    affines = filer.read_fits_affine_transform(date_stamp)
    svd_transforms = filer.read_fits_svd_transforms()
    cfg = svd_transforms[0]['configuration']
    # Load PSF library.  PSFs are sampled at 4x detector resolution so need to be down-sampled.
    ech_ord = cfg['ech_ord']
    psf_dict = toysim.load_psf_dict(opticon, ech_ord, downsample=True)
    _, psf_ext = psf_dict[0]
    oversampling = 4
    n_det_rows_slice = 256
    n_psf_rows_slice = n_det_rows_slice * oversampling
    n_psf_cols_slice = n_det_cols * oversampling
    slices_on_detector = {1: (15, 28), 2: (15, 28), 3: (1, 14), 4: (1, 14)}
    w_off = 0.01
    efp_xy_bs_detector = {1: (-xh, -w_off), 2: (+xh, +w_off), 3: (-xh, -w_off), 4: (+xh, +w_off)}
    tau_blaze = None

    opt_transforms = Util.find_optimum_transforms(w_cen, svd_transforms)
    cfg = opt_transforms[1]['configuration']
    ech_ord = cfg['ech_ord']
    w_blaze, tau_blaze = toysim.make_tau_blaze(ech_ord)  # Make echelle blaze profile for this order.

    mosaic = []                 # images are cartoons convolved with the PSF
    for det_no in range(1, 5):
        mosaic.append(np.zeros(det_shape))

    for slice_no in range(1, 29):
        # Start by mapping slice centre to detectors to get range of rows illuminated
        opt_transform = opt_transforms[slice_no]
        cfg = opt_transform['configuration']
        w_min, w_max = cfg['w_min'], cfg['w_max']
        yc = Util.slice_to_efp_y(slice_no, 0.)              # Slice y (beta) coordinate in EFP
        efp_y = np.array([yc, yc])
        efp_x = np.array([0., 0.])                        # Slice x (alpha) bounds in EFP, map to dfp_y
        efp_w = np.array([w_min, w_max])
        efp_slice = {'efp_y': efp_y, 'efp_x': efp_x, 'efp_w': efp_w}     # Centre of slice for detector
        mfp_slice = Util.efp_to_mfp(opt_transform, efp_slice)
        dfp_slice = Util.mfp_to_dfp(affines, mfp_slice)
        strip_row_min = int(dfp_slice['dfp_y'][0] - 60)      # Bracket slices which typically cover 120 rows..
        # Generate slice spanning image up-sampled to the psf resolution for
        strip_row_max = strip_row_min + n_det_rows_slice
        # Get pinhole PSF to apply to this slice (if a PSF exists)
        for det_no in dfp_slice['det_nos']:
            det_idx = det_no - 1

            ext_illum = np.zeros((n_det_rows_slice, n_det_cols))
            psf_illum = np.zeros((n_det_rows_slice, n_det_cols))

            dfp_det_nos = np.full(n_det_cols, det_no)
            dfp_pix_xs = np.arange(n_det_cols)  # Detector column indices

            n_rows_written = 0
            for det_row in range(strip_row_min, strip_row_max):
                dfp_pix_ys = np.full(n_det_cols, det_row)
                dfp_row = {'dfp_x': dfp_pix_xs, 'dfp_y': dfp_pix_ys, 'det_nos': dfp_det_nos}
                efp_row = Util.dfp_to_efp(opt_transform, affines, dfp_row)
                efp_x = efp_row['efp_x']
                idx_illum = np.argwhere(np.abs(efp_x) < xh)
                n_ib = len(idx_illum)
                if n_ib == 0:       # Skip unilluminated rows.
                    continue
                # Apply extended spectrum (sky or black body) to illuminated rows
                strip_row = det_row - strip_row_min
                efp_w_row = efp_row['efp_w']
                w_obs = efp_w_row[idx_illum][:]
                f_ext_obs = np.interp(w_obs, w_ext, f_ext)
                tau_ext_obs = np.interp(w_obs, w_blaze, tau_blaze)
                ext_illum[strip_row, idx_illum] = f_ext_obs * tau_ext_obs
                n_rows_written += 1

            psf_pnh = None
            if use_pinholes:
                slice_no_pnh_cen = pinhole['slice_no_cen']
                slice_no_offset = slice_no - slice_no_pnh_cen
                if math.fabs(slice_no_offset) < 5:          # Pinhole PSF exists for this slice.
                    _, psf_pnh = psf_dict[slice_no_offset]

            if psf_pnh is not None:     # Pinhole is visible in this slice.  Find rows it maps onto
                # Get pinhole trace
                efp_x_pnh = np.full(n_det_cols, pinhole['efp_x'])
                efp_y_pnh = np.full(n_det_cols, pinhole['efp_y'])
                efp_w_pnh = efp_w_row

                efp_pnh = {'efp_x': efp_x_pnh, 'efp_y': efp_y_pnh, 'efp_w': efp_w_pnh}
                dfp_pnh = Util.efp_to_dfp(opt_transform, affines, det_no, efp_pnh)
                # Row by row population of psf_illum image
                dfp_y_pnh = dfp_pnh['dfp_y']
                dfp_rows_pnh = np.round(dfp_y_pnh).astype(int)
                det_row_min_psf, det_row_max_psf = np.amin(dfp_rows_pnh), np.amax(dfp_rows_pnh)
                for det_row in range(det_row_min_psf, det_row_max_psf + 1):
                    dfp_pix_ys = np.full(n_det_cols, det_row)
                    idx_illum = np.argwhere(dfp_rows_pnh == det_row)
                    n_ib = len(idx_illum)
                    if n_ib == 0:  # Skip unilluminated rows.
                        continue
                    # Apply extended spectrum (sky or black body) to illuminated rows
                    strip_row = det_row - strip_row_min
                    w_obs = efp_w_row[idx_illum][:]
                    idx_min, idx_max = np.amin(idx_illum), np.amax(idx_illum)
                    w_min, w_max = np.amin(w_obs), np.amax(w_obs)
                    fmt = "Applying pnh spectrum to row= {:d}, cols= {:d}-{:d}, waves= {:8.5f}-{:8.5f}"
                    print(fmt.format(strip_row, idx_min, idx_max, w_min, w_max))
                    f_pnh_obs = np.interp(w_obs, w_pnh, f_pnh)
                    tau_pnh_obs = np.interp(w_obs, w_blaze, tau_blaze)
                    psf_illum[strip_row, idx_illum] = f_pnh_obs

            frame = mosaic[det_idx]
            # Now convolve cartoon slice with psf and then add dark current and read noise (Finger, Rauscher)
            image_slice_ext = scipy.signal.convolve2d(ext_illum, psf_ext, mode='same', boundary='symm')
            frame[strip_row_min:strip_row_max] += image_slice_ext
            if psf_pnh is not None:
                image_slice_psf = scipy.signal.convolve2d(psf_illum, psf_pnh, mode='same', boundary='symm')
                frame[strip_row_min:strip_row_max] += image_slice_psf

            t_now = time.perf_counter()
            t_min = (t_now - t_start) / 60.
            fmt = "\r- simulated, slice {:2d}, detector {:2d}, written {:d} rows at t= {:5.2f} min"  # \r at start
            print(fmt.format(slice_no, det_no, n_rows_written, t_min), end="", flush=True)

    image_out_path = '../output/' + sim_config + '.fits'
    filer.write_fits(image_out_path, hdr_primary, mosaic)

print()
print('lms_distort - Done')
