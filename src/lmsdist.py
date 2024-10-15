#!/usr/bin/env python
"""

"""
from os import listdir
import numpy as np
# from scipy.signal import decimate
from lms_filer import Filer
from lmsdist_util import Util
from lmsdist_plot import Plot
from lmsdist_trace import Trace
from lms_globals import Globals
from lms_detector import Detector
from lmsiq_image_manager import ImageManager

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
debug = True
if evaluate_transforms:

    print('lms_distort - Evaluating transforms')

    # Define EFP coordinates and target wavelength for spectrum
    efp_x_cen, efp_y_cen, efp_w_cen = 0., 0., 4.65
    tgt_slice_nos, test_phases = Util.efp_y_to_slice([efp_y_cen])
    tgt_slice_no = tgt_slice_nos[0]

    # Generate test spectrum for the wavelength/order which is closest to the mfp_y = 0. column.
    affines = filer.read_fits_affine_transform(date_stamp)
    svd_transforms = filer.read_fits_svd_transforms()
    opt_transform = Util.find_optimum_transform(tgt_slice_no, efp_w_cen, svd_transforms)

    # Check out and back for the wavelength min, max and centre of the spectrum.
    w_min = opt_transform['configuration']['w_min']
    w_max = opt_transform['configuration']['w_max']

    n_pts = 80
    # Create EFP data cube
    efp_ws = np.linspace(w_min, w_max, n_pts)
    efp_ys = np.zeros(n_pts)
    efp_as_mm = Globals.efp_as_mm
    alpha_fov = Globals.alpha_fov
    efp_xmax = alpha_fov / efp_as_mm
    xh = efp_xmax/2.
    efp_xs = np.linspace(-xh, +xh, n_pts)
    efp_out = {'efp_x': efp_xs, 'efp_y': efp_ys, 'efp_w': efp_ws}

    test_transforms = []

    config = opt_transform['configuration']
    slice_no = config['slice']
    matrices = opt_transform['matrices']
    mfp_points = Util.efp_to_mfp(opt_transform, efp_out)
    dfp_points = Util.mfp_to_dfp(affines, mfp_points)
    mfp_x, mfp_y = mfp_points['mfp_x'], mfp_points['mfp_y']
    efp_back = Util.mfp_to_efp(opt_transform, mfp_points)
    det_nos, dfp_x, dfp_y = dfp_points['det_nos'], dfp_points['dfp_x'], dfp_points['dfp_y']
    pri_ang = config['pri_ang']
    ech_ang = config['ech_ang']
    ech_ord = config['ech_ord']

    if debug:
        fmt = "{:>7s},{:>7s},{:>7s},{:>8s},{:>9s},{:>9s},{:>12s},{:>9s},{:>12s},{:>9s},{:>9s},{:>8s},{:>8s},{:>15s},{:>12s}"
        print(fmt.format('out', 'out', 'out', 'prism', 'echelle', 'echelle',
                         'mosaic', 'mosaic', 'det', 'det', 'det', 'back', 'back', 'back-out', 'back-out'))
        print(fmt.format('efp_x', 'efp_y', 'efp_w', 'angle', 'angle', 'order',
                         'mfp_x', 'mfp_y', 'no.', 'dfp_x', 'dfp_y', 'efp_x', 'efp_y', 'delta_efp_x', 'delta_efp_y'))
        print(fmt.format('mm', 'mm', 'micron', 'deg.', 'deg.', '-',
                         'mm', 'mm', '-', 'pix', 'pix', 'mm', 'mm', 'mm', 'mm'))
        fmt1 = "{:7.3f},{:7.3f},{:7.3f},{:8.3f},{:9.3f},{:9d},"
        fmt2 = "{:12.3f},{:9.3f},{:12d},{:9.1f},{:9.1f},{:8.3f},{:8.3f},{:15.3f},{:12.3f}"
        fmt = fmt1 + fmt2
        for i in range(0, n_pts):
            efp_out_x, efp_out_y = efp_out['efp_x'][i], efp_out['efp_y'][i]
            efp_back_x, efp_back_y = efp_back['efp_x'][i], efp_back['efp_y'][i]
            delta_efp_x = efp_back_x - efp_out_x
            delta_efp_y = efp_back_y - efp_out_y
            print(fmt.format(efp_out_x, efp_out_y, efp_out['efp_w'][i],
                             pri_ang, ech_ang, ech_ord,
                             mfp_x[i], mfp_y[i],
                             det_nos[i], dfp_x[i], dfp_y[i],
                             efp_back_x, efp_back_y,
                             delta_efp_x, delta_efp_y))

    # Read header and data shape (only) in from the template.
    data_exts = [1, 2, 3, 4]
    hdr_primary, data_list = filer.read_fits('../data/sim_template.fits', data_exts=data_exts)
    det_shape = data_list[0].shape
    det_imgs = []               # Initialise detector arrays with noise
    for i in range(4):
        det_imgs.append(np.random.normal(loc=1.0, scale=0.1, size=det_shape))

    # Read in PSFs for modelled slices bracketing a target centred on slice 12.
    oversampling = 4
    iq_date_stamp = '2024073000'
    iq_dataset_folder = '../data/iq/nominal/' + iq_date_stamp + '/'
    config_no = 41 - ech_ord
    iq_config_str = "_config{:03d}".format(config_no)
    iq_field_str = "_field{:03d}".format(1)
    iq_defoc_str = '_defoc000um'
    iq_config_str = 'lms_' + iq_date_stamp + iq_config_str + iq_field_str + iq_defoc_str
    # iq_folder = '../data/iq/nominal/' + iq_dataset + '/lms_2024073000_config020_field001_defoc000um/'
    iq_folder = iq_dataset_folder + iq_config_str + '/'
    amin, vmin, scale, hw_det_psf = None, None, None, None
    psf_dict = {}
    for slice_no in range(9, 18):
        iq_slice_str = "_spat{:02d}".format(slice_no) + '_spec0_detdesi'
        iq_filename = iq_config_str + iq_slice_str + '.fits'
        iq_path = iq_folder + iq_filename
        hdr, psf = filer.read_fits(iq_path)
        # print("slice_no={:d}, psf_max={:10.3e}".format(slice_no, np.amax(psf)))
        psf_dict[slice_no] = hdr, psf

    _, psf_tgt = psf_dict[13]
    _, n_psfcols = psf_tgt.shape
    # Rescale so that target PSF has 0. < v < 1.0 for (target) slice 13
    amin, amax = np.amin(psf_tgt), np.amax(psf_tgt)
    vmin, vmax = 0., 100.
    scale = (vmax - vmin) / (amax - amin)
    # Get PSF dimensions
    nr_psf, nc_psf = psf_tgt.shape
    hw_det_psf = nc_psf // (2 * oversampling)  # Half-width of PSF image on detector

    for slice_no in range(9, 18):
        # # Generate test spectrum for the wavelength/order which is closest to the mfp_y = 0. column.
        # affines = filer.read_fits_affine_transform(date_stamp)      # Get distortion transform for each slice
        # svd_transforms = filer.read_fits_svd_transforms()
        opt_transform = Util.find_optimum_transform(slice_no, efp_w_cen, svd_transforms)
        mfp_points = Util.efp_to_mfp(opt_transform, efp_out)
        dfp_points = Util.mfp_to_dfp(affines, mfp_points)

        _, psf = psf_dict[slice_no]
        print("slice_no={:d}, psf_max={:10.3e}".format(slice_no, np.amax(psf)))

        psf -= amin
        psf *= scale
        psf += vmin
        print("slice_no={:d}, psf_max={:10.3e}".format(slice_no, np.amax(psf)))

        fw_det_psf = 2 * hw_det_psf
        for i in range(0, n_pts):
            det_idx = dfp_points['det_nos'][i] - 1
            det_x = dfp_points['dfp_x'][i]
            det_y = dfp_points['dfp_y'][i]
            if det_x < 0. or det_x > 2048.:
                continue
            det_img = det_imgs[det_idx]
            _, n_imcols = det_img.shape
            # Add the PSF to the image at this location at the PSF resolution (x4 image pix/det pix)
            r1, c1 = int(det_y - hw_det_psf), int(det_x - hw_det_psf)
            c1 = 0 if c1 < 0 else c1
            r2, c2 = r1 + fw_det_psf, c1 + fw_det_psf
            pc1, pc2 = 0, n_psfcols
            if c1 >= n_imcols or c2 < 1:
                continue
            if c1 < 0:
                pc1 = -c1 * oversampling
                c1 = 0
            if c2 > n_imcols:
                ncr = c2 - n_imcols
                pc2 = n_psfcols - ncr * oversampling
                c2 = n_imcols
            # det_patch = np.array(det_img[r1:r2, c1:c2])
            # up-sample detector patch and add psf
            # det_patch_us1 = np.repeat(det_patch, oversampling, axis=0)
            # det_patch_us = np.repeat(det_patch_us1, oversampling, axis=1)
            # det_patch_us += psf[:, pc1:pc2]
            # down-sample patch and write back into image
            psf_sub = psf[:, pc1:pc2]
            nss_rows = psf_sub.shape[0] // oversampling
            nss_cols = psf_sub.shape[1] // oversampling
            psf_ds = psf_sub.reshape((nss_rows, oversampling, nss_cols, -1)).mean(axis=3).mean(axis=1)
            det_imgs[det_idx][r1:r2, c1:c2] += psf_ds

    out_path = '../output/stimage.fits'
    filer.write_fits(out_path, hdr_primary, det_imgs)

print('lms_distort - Done')
