from lmsiq_analyse import LMSIQAnalyse
from lmsiq_filer import LMSIQFiler
from lmsiq_zemaxio import LMSIQZemaxio
from lmsiq_plot import LMSIQPlot
from lmsiq_shift import LMSIQShift
from lms_util import Util
from lms_wcal import Wcal
from lms_globals import Globals
from lms_ipc import Ipc
from lms_ipg import Ipg
from lms_detector import Detector
import numpy as np

print('LMS Repeatability (lmsrep.py) - started')

det_pix_size = Globals.mm_lmspix * 1000.
run_dict = {'20221014': 11, '20221026': 21}
dataset = '20221026'            # 'psf_model_20221014_multi_wavelength'
tol_number = 0

fmt = "Analysing repeatability impact of run dataset {:s}, Tol= {:d}"
print(fmt.format(dataset, tol_number))

filer = LMSIQFiler(dataset)
zemaxio = LMSIQZemaxio()
analyse = LMSIQAnalyse()
plot = LMSIQPlot()
shift = LMSIQShift()
util = Util()
wcal = Wcal()
detector = Detector()
wcal.read_poly()

add_ipc = True         # True = Add Inter Pixel Capacitance crosstalk (1.3 % - Rauscher ref.)
ipc_factor_nominal = 0.013
ipc_factor = 0.013

print("IPC modelling included = {:s}".format(str(add_ipc)))
if add_ipc:
    fmt = " - using IPC factor = {:10.3f} (Note nominal value is {:10.3f}"
    print(fmt.format(ipc_factor, ipc_factor_nominal))

add_ipg = True         # True = Add Inter Pixel Capacitance crosstalk (1.3 % - Rauscher ref.)
ipg = None
print("Intra Pixel Gain modelling included = {:s}".format(str(add_ipg)))

plot_images, plot_profiles = False, False
print("Plotting images =   {:s}".format(str(plot_images)))
print("Plotting profiles = {:s}".format(str(plot_profiles)))

run_test = False         # Generate test case with 50 % IPC and an image with one bright pixel.
print("Running test case = {:s}".format(str(run_test)))

ifu_slice = 1
location = 'middle'
axes = ['spectral', 'spatial']
poly_file = '../output/lms_dist_poly_old.txt'           # Distortion polynomial

poly, ech_bounds = util.read_polyfits_file(poly_file)   # Use distortion map for finding dispersion
d_tel = 39.0E9          # ELT diameter in microns

n_runs = run_dict[dataset]
for folder_tag in range(0, n_runs):

    tag = "{:02d}".format(folder_tag)
    zemax_folder = dataset + '_current_tol_' + tag
    # For first dataset, read in parameters from text file and generate IPC and IPG kernels.
    configuration = zemaxio.read_param_file(dataset, zemax_folder)
    _, wave, _, _, order, im_pix_size = configuration
    im_oversampling = int(det_pix_size / im_pix_size)
    ipc = Ipc(ipc_factor, im_oversampling)
    ipg = Ipg(im_oversampling, 14, 0.44, 14, 0.44)  # Gain profiles, same in x and y
    if plot_images:
        plot_title = zemax_folder + ' Kernel'
        params = plot_title, None
        im_kernel, im_oversampling, det_kernel_size = Ipc.kernel
        title = "Inter Pixel Capacitance ({:6.3f} %)".format(ipc_factor)
        plot.images([(im_kernel, params)], nrowcol=(1, 1), title=title)
        ipg_image, _, _, _, _, _ = Ipg.kernel
        params = 'Intra-pixel gain kernel', 1.0
        plot.images([(ipg_image, params)], nrowcol=(1, 1))

    # Calculate dispersion and centre wavelength for each order.
    transform = util.get_polyfit_transform(poly, ech_bounds, configuration)
    dw_lmspix = util.find_dispersion(transform, configuration)

    # Select file to analyse
    filter_tags = ['0000', 'fits']      # Use the first M-C instance in each folder
    path, file_list = zemaxio.read_file_list(dataset, zemax_folder, filter_tags=filter_tags)
    file_name = file_list[0]
    col_perfect, col_design = 0, 1
    file_id = file_name[0:-5].split('_')[4]
    obs_1 = zemaxio.load_observation(path, file_name, '')
    obs_2 = ipc.convolve(obs_1)
    if plot_images:
        title = file_name
        plot.images([obs_1, obs_2], nrowcol=(1, 2),
                    title=title, pane_titles=['Pre-IPC', 'Post-IPC'], plotregion='centre')

    det_shift = -2.0           # Spectral shift in detector pixels
    det_shift_end = +0.0
    det_shift_increment = 2.0

    while det_shift <= det_shift_end:
        im_shift = det_shift * im_oversampling
        obs_3 = shift.sub_pixel_shift(obs_2, 'spectral', im_shift)
        title = zemax_folder
        obs_ratio = 100 * obs_3[0] / obs_2[0], obs_2[1]
        post_label = "Post-shift by {:4.1f} pix.".format(det_shift)
        if plot_images:
            plot.images([obs_2, obs_3], nrowcol=(2, 1), plotregion='centre',
                        title=title, pane_titles=['Pre-Shift', post_label])
            plot.images([obs_ratio], nrowcol=(1, 1), plotregion='centre',
                        title=title, pane_titles=[post_label])

        obs_4 = ipg.imprint(obs_3)      # Imprint IPG and detect (rebin to detector pixel scale).
        obs_5 = detector.measure(obs_4)
        plot.images([obs_2, obs_3, obs_4, obs_5], nrowcol=(2, 2), plotregion='all',
                    title=title, pane_titles=['Pre-Shift', post_label, 'Post-IPG', 'Frame'])

        det_shift += det_shift_increment

# Generate wavelength dependent summary file from profiles.
summary = filer.create_summary_header(axes)
print(summary)
folder_tag = 0
n_runs = run_dict[dataset]
for run in range(0, n_runs):
    tag = "{:02d}".format(run)
    folder = dataset + '_current_tol_' + tag
    configuration = zemaxio.read_param_file(dataset, folder)
    _, wave, _, _, order, im_pix_size = configuration
    # Calculate dispersion and centre wavelength for each order.
    transform = util.get_polyfit_transform(poly, ech_bounds, configuration)
    dw_lmspix = util.find_dispersion(transform, configuration)
    result = ''
    #    slice, wave, prism_angle, grating_angle, order, im_pix_size = filer.read_param_file(dataset, folder)
    strehls, strehl_errs = [], []
    for axis in axes:

        profile_folder = folder + '_profile'
        data_id = dataset, tag, axis
        ees_data, _, _ = filer.read_profiles(data_id, 'ee')
        lsf_data, strehl_data, ipc_factor = filer.read_profiles(data_id, 'lsf')

        strehl, strehl_err = strehl_data
        strehls.append(strehl)
        strehl_errs.append(strehl_err)

        folder, axis, xlms, lsf_mean, lsf_rms, lsf_all = ees_data

        im_oversampling = int(det_pix_size / im_pix_size)
        # Calculate dispersion and centre wavelength for each order.
        transform = util.get_polyfit_transform(poly, ech_bounds, configuration)
        dw_lmspix = util.find_dispersion(transform, configuration)

        x_ref, ee_axis_refs = analyse.find_ee_axis_references(wave, axis, xlms, lsf_mean, lsf_all)
        result += "{:>10.2f},".format(x_ref)
        for ee_val in ee_axis_refs:
            result += "{:>12.6f},".format(ee_val)

        if plot_profiles:
            plot.plot_ee(ees_data, wave, x_ref, ee_axis_refs, ipc_factor,  plot_all=True)
        if plot_profiles:
            plot.plot_lsf(lsf_data, wave, dw_lmspix, ipc_factor,  hwlim=6.0, plot_all=True)
        _, _, x, y_mean, y_rms, y_all = lsf_data
        ynorm = np.amax(y_mean)        # Normalise to peak of LSF
        # We find the error on the mean FWHM and its error by finding the FWHM of all profiles...
        xfwhm_list = []

        nr, nc = y_all.shape
        for c in range(0, nc):
            y = y_all[:, c]
            xl, xr, yh = LMSIQPlot.find_hwhm(x, y)
            xfwhm_list.append(xr - xl)
        xfwhms = np.array(xfwhm_list)
        xfwhm_ave_all = np.mean(xfwhms)
        xfwhm_err_all = np.std(xfwhms)
        result += "{:>15.6f},".format(xfwhm_ave_all)
        result += "{:>15.6f},".format(xfwhm_err_all)
        wfwhm = dw_lmspix * xfwhm_ave_all
        if axis == 'spectral':
            srp = wave / wfwhm
            wfwhm_err = dw_lmspix * xfwhm_err_all
            srp_err = srp * wfwhm_err / wfwhm
    fmt = "{:10.6f},{:8d},{:>8.3f},{:>10.0f},{:>10.0f},{:>12.6f},{:>12.6f},"
    pre_result = fmt.format(wave, order, ipc_factor, srp, srp_err, strehl, strehl_err)
    result = pre_result + result
    print(result)
    summary.append(result)
ipc_pc = 100. * ipc_factor
ipc_int = int(ipc_pc)
ipc_dec = int(1000. * (ipc_pc - ipc_int) + 0.5)
summary_id = "_ipc_{:02d}_{:03d}".format(ipc_int, ipc_dec)
filer.write_summary(dataset, summary, summary_id)

id_plot_list = ['_ipc_13_000', '_ipc_03_900', '_ipc_01_300', '_ipc_00_000']
profile_list = []
for id_plot in id_plot_list:
    profile = filer.read_summary(dataset, id_plot)
    if profile is not None:
        profile_list.append(profile)
plot.profiles(profile_list, config='strehl', ylabel='Strehl')
plot.profiles(profile_list, config='srp', ylabel='SRP')
plot.profiles(profile_list, config='fwhmspat', ylabel='FWHM spatial')
plot.profiles(profile_list, config='fwhmspec', ylabel='FWHM spectral')

print('LMS IQ - done')
