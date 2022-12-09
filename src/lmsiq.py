from lmsiq_analyse import LMSIQAnalyse
from lmsiq_filer import LMSIQFiler
from lmsiq_plot import LMSIQPlot
from lms_util import Util
from lms_wcal import Wcal
from lms_globals import Globals
from lms_ipc import Ipc
import numpy as np

print('LMS IQ - started')

reanalyse = True
print("Re-analysing from scratch = {:s}".format(str(reanalyse)))
add_ipc = True         # True = Add Inter Pixel Capacitance crosstalk (1.3 % - Rauscher ref.)
nom_ipc = 0.013
print("IPC modelling included = {:s}".format(str(add_ipc)))
if add_ipc:
    print(" Nominal IPC factor = {:10.3f}".format(nom_ipc))

plot_images, plot_profiles = False, False
print("Plotting images =   {:s}".format(str(plot_images)))
print("Plotting profiles = {:s}".format(str(plot_profiles)))

run_test = False         # Generate test case with 50 % IPC and an image with one bright pixel.
print("Running test case = {:s}".format(str(run_test)))
filer = LMSIQFiler()
analyse = LMSIQAnalyse()
plot = LMSIQPlot()
util = Util()
wcal = Wcal()
wcal.read_poly()

det_pix_size = Globals.mm_lmspix * 1000.
configs = {'20221014': 11, '20221026': 21}
dataset = '20221026'            # 'psf_model_20221014_multi_wavelength'

slice = 1
location = 'middle'
axes = ['spectral', 'spatial']
poly_file = '../output/lms_dist_poly_old.txt'           # Distortion polynomial

poly, ech_bounds = util.read_polyfits_file(poly_file)   # Use distortion map for finding dispersion
d_tel = 39.0E9          # ELT diameter in microns

fmt = "{:>10s},{:>8s},{:>8s},{:>10s},{:>10s},{:>12s},{:>12s},"
hdr1 = fmt.format('Wave', 'Order', 'IPC', 'SRP', 'SRP_err', 'Strehl', 'Strehl_err')
hdr2 = fmt.format('(um)', '-', '-', '-', '-', '-', '-')
fmt = "{:>10s},{:>12s},{:>12s},{:>12s},{:>15s},{:>15s},"
for axis in axes:
    ee_tag = 'EE' + axis[0:4]
    fwhm_tag = 'FWHM' + axis[0:4]
    hdr1 += fmt.format('X_' + ee_tag, ee_tag, ee_tag, ee_tag, fwhm_tag, fwhm_tag + 'err',)
    hdr2 += fmt.format('pix.', 'perfect', 'design', '<model>', 'pix.', 'pix.')
results = [hdr1, hdr2]


if reanalyse:
    print("Re-analysing dataset {:s}".format(dataset))

    folder_tag = 0
    n_folders = configs[dataset]
    for folder_tag in range(0, n_folders):
        tag = "{:02d}".format(folder_tag)
        folder = dataset + '_current_tol_' + tag
        folder_tag += 1
        # Read in parameters from text file
        print("Re-analysing all fits images in folder {:s}".format(folder))
        configuration = filer.read_param_file(dataset, folder)
        _, wave, _, _, order, im_pix_size = configuration
        oversampling = int(det_pix_size / im_pix_size)
        ipc_factor = nom_ipc if add_ipc else 0.0
        ipc = Ipc(ipc_factor, oversampling)
        if plot_images:
            title = folder + ' Kernel'
            params = title, None
            im_kernel, oversampling, det_kernel_size = Ipc.kernel
            plot.images([(im_kernel, params)], nrowcol=(1, 1), title=title)

        # Calculate dispersion and centre wavelength for each order.
        transform = util.get_polyfit_transform(poly, ech_bounds, configuration)
        dw_lmspix = util.find_dispersion(transform, configuration)

        for axis in axes:
            # Select all fits files in directory
            path, file_list = filer.read_file_list(dataset, folder)
            col_perfect, col_design = 0, 1
            observations = analyse.load_observations(path, file_list, run_test)
            if plot_images:
                title = folder + ' Pre-IPC'
                plot.images(observations[0:6], nrowcol=(2, 3), title=title)
            if add_ipc:
                for i, obs in enumerate(observations):
                    observations[i] = ipc.add_ipc(obs)
            if plot_images:
                title = folder + ' Post-IPC'
                plot.images(observations[0:6], nrowcol=(2, 3), title=title)
            ees_data = analyse.eed(observations, axis,
                                   log10sampling=True, normalise='to_average')
            lsf_data = analyse.lsf(observations, axis)
            strehl_data = analyse.strehl(observations)
            filer.write_profiles(folder, 'ee', axis, ees_data, strehl_data, ipc.factor)
            filer.write_profiles(folder, 'lsf', axis, lsf_data, strehl_data, ipc.factor)

print(hdr1)
print(hdr2)
folder_tag = 0
n_folders = configs[dataset]
for folder_tag in range(0, n_folders):
    tag = "{:02d}".format(folder_tag)
    folder = dataset + '_current_tol_' + tag
    folder_tag += 1
    configuration = filer.read_param_file(dataset, folder)
    _, wave, _, _, order, im_pix_size = configuration
    # Calculate dispersion and centre wavelength for each order.
    transform = util.get_polyfit_transform(poly, ech_bounds, configuration)
    dw_lmspix = util.find_dispersion(transform, configuration)
    result = ''
    #    slice, wave, prism_angle, grating_angle, order, im_pix_size = filer.read_param_file(dataset, folder)
    strehls, strehl_errs = [], []
    for axis in axes:

        ees_data, _, _ = filer.read_profiles(folder, 'ee', axis)
        lsf_data, strehl_data, ipc_factor = filer.read_profiles(folder, 'lsf', axis)

        strehl, strehl_err = strehl_data
        strehls.append(strehl)
        strehl_errs.append(strehl_err)

        folder, axis, xlms, lsf_mean, lsf_rms, lsf_all = ees_data

        oversampling = int(det_pix_size / im_pix_size)
        # Calculate dispersion and centre wavelength for each order.
        transform = util.get_polyfit_transform(poly, ech_bounds, configuration)
        dw_lmspix = util.find_dispersion(transform, configuration)

        x_ref, ee_axis_refs = analyse.find_ee_axis_references(wave, axis, xlms, lsf_mean, lsf_all)
        result += "{:>10.2f},".format(x_ref)
        for ee_val in ee_axis_refs:
            result += "{:>12.6f},".format(ee_val)

        if plot_profiles:
            plot.plot_ee(ees_data, wave, x_ref, ee_axis_refs, ipc_factor,  plot_all=True)
        lsf_data, _, _ = filer.read_profiles(folder, 'lsf', axis)
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
    results.append(result)
ipc_pc = 100. * ipc_factor
ipc_int = int(ipc_pc)
ipc_dec = int(1000. * (ipc_pc - ipc_int) + 0.5)
summary_id = "_ipc_{:02d}_{:03d}".format(ipc_int, ipc_dec)
filer.write_summary(dataset, results, summary_id)

id_plot_list = ['_ipc_13_000', '_ipc_03_900', '_ipc_01_300', '_ipc_00_000']

profile_list = []
for id_plot in id_plot_list:
    profile = filer.read_summary(dataset, id_plot)
    profile_list.append(profile)
plot.profiles(profile_list, config='strehl', ylabel='Strehl')
plot.profiles(profile_list, config='srp', ylabel='SRP')
plot.profiles(profile_list, config='fwhmspat', ylabel='FWHM spatial')
plot.profiles(profile_list, config='fwhmspec', ylabel='FWHM spectral')

print('LMS IQ - done')
