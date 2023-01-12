from lmsiq_analyse import Analyse
from lmsiq_filer import Filer
from lmsiq_zemaxio import LMSIQZemaxio
from lmsiq_plot import LMSIQPlot
from lms_util import Util
from lms_wcal import Wcal
from lms_ipc import Ipc
from lms_ipg import Ipg
from lms_detector import Detector
import numpy as np

print('LMS Image Quality (lmsiq.py) - started')

detector = Detector()
#det_pix_size = Globals.mm_lmspix * 1000.
nconfigs_dict = {'20221014': (11, 10), '20221026': (21, 100)}
dataset = '20221026'            # 'psf_model_20221014_multi_wavelength'

n_configs, n_mcruns = nconfigs_dict[dataset]
# n_configs = 2
# n_mcruns = 4
n_mcruns_tag = "{:04d}".format(n_mcruns)

filer = Filer(dataset)
zemaxio = LMSIQZemaxio()
analyse = Analyse()
plot = LMSIQPlot()
util = Util()
wcal = Wcal()
wcal.read_poly()

ipg, ipc, folder_tag = None, None, ''

reanalyse = True
print("Re-analysing from scratch = {:s}".format(str(reanalyse)))
add_ipc = True         # True = Add Inter Pixel Capacitance crosstalk (1.3 % - Rauscher ref.)
ipc_factor_nominal = 0.013
ipc_factor = 0.013

print("IPC modelling included = {:s}".format(str(add_ipc)))
if add_ipc:
    fmt = " - using IPC factor = {:10.3f} (Note nominal value is {:10.3f}"
    print(fmt.format(ipc_factor, ipc_factor_nominal))

add_ipg = False         # True = Add Inter Pixel Capacitance crosstalk (1.3 % - Rauscher ref.)
print("Intra Pixel Gain modelling included = {:s}".format(str(add_ipg)))

plot_images, plot_profiles = False, False
print("Plotting images =   {:s}".format(str(plot_images)))
print("Plotting profiles = {:s}".format(str(plot_profiles)))

run_test = False         # Generate test case with 50 % IPC and an image with one bright pixel.
print("Running test case = {:s}".format(str(run_test)))

flat_test = False            # Write a unity flat field into the Zemax data

ifu_slice = 1
location = 'middle'
axes = ['spectral', 'spatial']
poly_file = '../output/lms_dist_poly_old.txt'           # Distortion polynomial

poly, ech_bounds = util.read_polyfits_file(poly_file)   # Use distortion map for finding dispersion
d_tel = 39.0E9          # ELT diameter in microns

if reanalyse:
    print("Analysing image quality for dataset {:s}".format(dataset))
    first_config = True
    configuration, ipc = None, None
    for config_number in range(0, n_configs):
        tag = "{:02d}".format(config_number)
        zemax_folder = dataset + '_current_tol_' + tag
        fmt = "\r- Config {:02d} of {:d}"
        print(fmt.format(config_number, n_configs), end="", flush=True)

        if first_config:
            # For first dataset, read in parameters from text file and generate IPC and IPG kernels.
            configuration = zemaxio.read_param_file(dataset, zemax_folder)
            _, wave, _, _, order, im_pix_size = configuration
            im_oversampling = int(Detector.det_pix_size / im_pix_size)
            ipc = Ipc(ipc_factor, im_oversampling)
            if add_ipc:
                ipc = Ipc(ipc_factor, im_oversampling)
                folder_tag += ipc.get_tag(ipc_factor)
            if add_ipg:
                ipg = Ipg(im_oversampling, 14, 0.44, 14, 0.44)  # Gain profiles, same in x and y
                folder_tag += '_ipg'
            if plot_images:
                if add_ipc:
                    plot_title = zemax_folder + ' Kernel'
                    params = plot_title, None
                    im_kernel, im_oversampling, det_kernel_size = Ipc.kernel
                    title = "Diffusion + IPC ({:6.3f} %) kernel".format(ipc_factor)
                    plot.images([(im_kernel, params)], nrowcol=(1, 1), title=title)
                if add_ipg:
                    ipg_image, _, _, _, _, _ = Ipg.kernel
                    params = 'Intra-pixel gain kernel', 1.0
                    plot.images([(ipg_image, params)], nrowcol=(1, 1))
            first_config = False

        # Calculate dispersion and centre wavelength for each order.
        transform = util.get_polyfit_transform(poly, ech_bounds, configuration)
        dw_lmspix = util.find_dispersion(transform, configuration)

        for axis in axes:
            # Select all fits files in directory
            path, file_list = zemaxio.read_file_list(dataset, zemax_folder, filter_tags=['fits'])
            col_perfect, col_design = 0, 1
            observations = zemaxio.load_dataset(path, file_list)

            if plot_images:
                title = zemax_folder + ' Pre-IPC'
                plot.images(observations[0:6], nrowcol=(2, 3), plotregion='centre',
                            title=title, pane_titles='file_id')
            if add_ipc:
                for i, obs in enumerate(observations):
                    if flat_test:
                        obs = detector.set_flat(obs)
                    observations[i] = ipc.convolve(obs)
            if add_ipg:
                for i, obs in enumerate(observations):
                    observations[i] = ipg.imprint(obs)
            if plot_images:
                title = zemax_folder + ' Post-IPC'
                plot.images(observations[0:6],
                            nrowcol=(2, 3), plotregion='centre',
                            title=title, pane_titles='file_id')
            ees_data = analyse.eed(observations, axis,
                                   log10sampling=True, normalise='to_average')
            lsf_data = analyse.lsf(observations, axis, v_coadd=12.0, u_radius=20.0)
            strehl_data = analyse.strehl(observations)
            filer.write_profiles((dataset, tag, axis), ees_data, strehl_data, ipc_factor, 'ee')
            filer.write_profiles((dataset, tag, axis), lsf_data, strehl_data, ipc_factor, 'lsf')

    # Generate wavelength dependent summary file from profiles.
    summary = filer.create_summary_header(axes)
    print(summary)
#    config_number = 0
#    n_runs = nconfigs_dict[dataset]
    for config_number in range(0, n_configs):
        tag = "{:02d}".format(config_number)
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

#            profile_folder = folder + '_profile'
            data_id = dataset, tag, axis
            ees_data, _, _ = filer.read_profiles(data_id, 'ee')
            lsf_data, strehl_data, ipc_factor = filer.read_profiles(data_id, 'lsf')

            strehl, strehl_err = strehl_data
            strehls.append(strehl)
            strehl_errs.append(strehl_err)

            folder, axis, xlms, lsf_mean, lsf_rms, lsf_all = ees_data

            im_oversampling = Detector.det_pix_size / im_pix_size
            # Calculate dispersion and centre wavelength for each order.
#            transform = util.get_polyfit_transform(poly, ech_bounds, configuration)
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
    summary_tag = ipc.get_tag(ipc_factor)
    filer.write_summary(dataset, summary, summary_tag)

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
