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
run_tests = False
print("Running tests = {:s}".format(str(run_tests)))
print("- generate high resolution column avaregaed diffusion profile ")
if run_tests:
    ipc_test_factor, ipc_test_oversampling = 0.0, 25
    ipc_test = Ipc(ipc_test_factor, ipc_test_oversampling)
    ipc_test.plot_kernel_profile('Test kernel profile', max_val=3000)

detector = Detector()
config_dict = {'20221026': (21, 100, '_current_tol_', 'flat M12'),
               '20230207': (11, 100, '_with_toroidalM12_', 'toroidal M12, slice  1'),
               '20230208': (21, 100, '_with_toroidalM12_', 'toroidal M12, slice 14')
               }
dataset = '20230208'

n_configs, n_mcruns, folder_name, plot_label = config_dict[dataset]
n_mcruns_tag = "{:04d}".format(n_mcruns)

filer = Filer(dataset)
zemaxio = LMSIQZemaxio()
analyse = Analyse()
plot = LMSIQPlot()
util = Util()
wcal = Wcal()
wcal.read_poly()

ipg, folder_tag = None, ''

boo_fmt = "{:40s}= {:s}"
reanalyse = False
print(boo_fmt.format('Re-analysing from scratch', str(reanalyse)))
inter_pixel = True              # True = Add Inter Pixel Capacitance crosstalk (1.3 % - Rauscher ref.) and diffusion.
ipc_factor_nominal = 0.013
ipc_factor = 0.013

print(boo_fmt.format('IPC and diffusion modelling included', str(inter_pixel)))
if inter_pixel:
    fmt = " - using IPC factor = {:10.3f} (Note nominal value is {:10.3f})"
    print(fmt.format(ipc_factor, ipc_factor_nominal))

add_ipg = False                 # True = Add Inter Pixel Capacitance crosstalk (1.3 % - Rauscher ref.)
print(boo_fmt.format('Intra Pixel Gain modelling included', str(add_ipg)))

run_test = False                # Generate test case with 50 % IPC and an image with one bright pixel.
print(boo_fmt.format('Running test case', str(run_test)))
plot_images, plot_profiles = False, False
print(boo_fmt.format('Plotting images', str(plot_images)))
print(boo_fmt.format('Plotting profiles', str(plot_profiles)))

flat_test = False               # Write a unity flat field into the Zemax data

ifu_slice = 1
location = 'middle'
axes = ['spectral', 'spatial']
poly_file = '../output/lms_dist_poly_old.txt'           # Distortion polynomial

poly, ech_bounds = util.read_polyfits_file(poly_file)   # Use distortion map for finding dispersion
d_tel = 39.0E9                  # ELT diameter in microns

if reanalyse:
    print("Analysing image quality for dataset {:s}".format(dataset))
    first_config = True
    configuration, ipc = None, None
    for config_number in range(0, n_configs):
        config_tag = "{:02d}".format(config_number)
        zemax_folder = dataset + folder_name + config_tag
        fmt = "\r- Config {:02d} of {:d}"
        print(fmt.format(config_number, n_configs), end="", flush=True)

        if first_config:
            # For first dataset, read in parameters from text file and generate IPC and IPG kernels.
            configuration = zemaxio.read_param_file(dataset, zemax_folder)
            _, wave, _, _, order, im_pix_size = configuration
            im_oversampling = int(Detector.det_pix_size / im_pix_size)
            if inter_pixel:
                ipc = Ipc(ipc_factor, im_oversampling)
                folder_tag += ipc.get_tag(ipc_factor)
            if add_ipg:
                ipg = Ipg(im_oversampling, 14, 0.44, 14, 0.44)  # Gain profiles, same in x and y
                folder_tag += '_ipg'
            if plot_images:
                if inter_pixel:
                    im_kernel = Ipc.kernel
                    title = "Diffusion + IPC ({:6.3f} %) kernel".format(ipc_factor)
                    params = title, 1.0
                    plot.images([(im_kernel, params)], nrowcol=(1, 1), title=title, do_log=False)
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
            if inter_pixel:
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
            data_id = dataset, folder_tag, config_tag, n_mcruns_tag, axis

            filer.write_profiles(data_id, ees_data, strehl_data, ipc_factor, 'ee')
            filer.write_profiles(data_id, lsf_data, strehl_data, ipc_factor, 'lsf')

    # Generate wavelength dependent summary file from profiles.
    summary = filer.create_summary_header(axes)
    print(summary)
#    config_number = 0
#    n_runs = nconfigs_dict[dataset]
    for config_number in range(0, n_configs):
        config_tag = "{:02d}".format(config_number)
        folder = dataset + folder_name + config_tag
        configuration = zemaxio.read_param_file(dataset, folder)
        _, wave, _, _, order, im_pix_size = configuration
        # Calculate dispersion and centre wavelength for each order.
        transform = util.get_polyfit_transform(poly, ech_bounds, configuration)
        dw_lmspix = util.find_dispersion(transform, configuration)
        result = ''
        #    slice, wave, prism_angle, grating_angle, order, im_pix_size = filer.read_param_file(dataset, folder)
        strehls, strehl_errs = [], []
        srp, srp_err, strehl, strehl_err = None, None, None, None
        for axis in axes:
            data_id = dataset, folder_tag, config_tag, n_mcruns_tag, axis
            ees_data, _, _ = filer.read_profiles(data_id, 'ee')
            lsf_data, strehl_data, ipc_factor = filer.read_profiles(data_id, 'lsf')

            strehl, strehl_err = strehl_data
            strehls.append(strehl)
            strehl_errs.append(strehl_err)

            folder, axis, xlms, lsf_mean, lsf_rms, lsf_all = ees_data

            im_oversampling = Detector.det_pix_size / im_pix_size
            # Calculate dispersion and centre wavelength for each order.
            dw_lmspix = util.find_dispersion(transform, configuration)

            x_ref, ee_axis_refs = analyse.find_ee_axis_references(wave, axis, xlms, lsf_mean, lsf_all)
            result += "{:>10.2f},".format(x_ref)
            for ee_val in ee_axis_refs:
                result += "{:>12.6f},".format(ee_val)
            if plot_profiles:
                plot.plot_ee(ees_data, wave, x_ref, ee_axis_refs, ipc_factor, plot_all=True)
            if plot_profiles:
                plot.plot_lsf(lsf_data, wave, dw_lmspix, ipc_factor, hwlim=6.0, plot_all=True)
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
    summary_tag = ''
    if ipc is not None:
        summary_tag = ipc.get_tag(ipc_factor)
    filer.write_summary(dataset, summary, summary_tag)

# The following sections plot data generated in *.csv files when running with doanalysis = True
# 1. Plot Zemax configuration data (grating/prism angles etc.)
configurations = []
for config_number in range(0, n_configs):
    config_tag = "{:02d}".format(config_number)
    folder = dataset + folder_name + config_tag
    configuration = zemaxio.read_param_file(dataset, folder)
    configurations.append(configuration)

# plot.configurations(configurations)

plot_id_list = [('20221026', ''), ('20221026', '_ipc_01_3'),
                ('20230207', ''), ('20230207', '_ipc_01_3'),
                ('20230208', ''), ('20230208', '_ipc_01_3')]
profile_list = []
for plot_id in plot_id_list:
    dataset, tag = plot_id
    profile = filer.read_summary(dataset, tag)
    if profile is not None:
        profile_list.append(profile)
plot.profiles(profile_list, config_dict, config='strehl', ylabel='Strehl')
plot.profiles(profile_list, config_dict, config='srp', ylabel='SRP')
plot.profiles(profile_list, config_dict, config='fwhmspat', ylabel='FWHM spatial')
plot.profiles(profile_list, config_dict, config='fwhmspec', ylabel='FWHM spectral')

print('LMS IQ - done')
