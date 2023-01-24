from lmsiq_analyse import Analyse
from lmsiq_filer import Filer
from lmsiq_zemaxio import LMSIQZemaxio
from lmsiq_plot import LMSIQPlot
from lmsiq_shift import LMSIQShift
from lms_util import Util
from lms_wcal import Wcal
from lms_ipc import Ipc
from lms_ipg import Ipg
from lms_detector import Detector
import numpy as np

print('LMS Repeatability (lmsrep.py) - started')
print()

detector = Detector()

nconfigs_dict = {'20221014': (11, 10), '20221026': (21, 100)}
dataset = '20221026'            # 'psf_model_20221014_multi_wavelength'
tol_number = 0

reanalyse = False
print("Reanalysing LSF centroid shifts = {:s}".format(str(reanalyse)))

filer = Filer(dataset)
zemaxio = LMSIQZemaxio()
analyse = Analyse()
plot = LMSIQPlot()
shift = LMSIQShift()
util = Util()
wcal = Wcal()

wcal.read_poly()

add_ipc = False         # True = Add Inter Pixel Capacitance crosstalk (1.3 % - Rauscher ref.)
ipc = None
ipc_factor_nominal = 0.013
ipc_factor = 0.013

print("Inter Pixel Capacitance modelling included = {:s}".format(str(add_ipc)))
if add_ipc:         # True = Add Inter Pixel Capacitance crosstalk (1.3 % - Rauscher ref.)
    fmt = " - using IPC factor = {:10.3f} (Note nominal value is {:10.3f}"
    print(fmt.format(ipc_factor, ipc_factor_nominal))

add_ipg = False
ipg, ipc, folder_tag = None, None, ''
im_oversampling = None
ndet_rows, ndet_cols = None, None
print("Intra Pixel Gain modelling included = {:s}".format(str(add_ipg)))

plot_images, plot_profiles = False, True
print("Plotting images =   {:s}".format(str(plot_images)))
print("Plotting profiles = {:s}".format(str(plot_profiles)))

run_test = False         # Generate test case with 50 % IPC and an image with one bright pixel.
print("Running test case = {:s}".format(str(run_test)))

ifu_slice = 1
location = 'middle'
axes = ['spectral', 'spatial']
axis = 'spectral'
poly_file = '../output/lms_dist_poly_old.txt'           # Distortion polynomial

poly, ech_bounds = util.read_polyfits_file(poly_file)   # Use distortion map for finding dispersion
d_tel = 39.0E9          # ELT diameter in microns

n_configs, n_mcruns = nconfigs_dict[dataset]
#n_configs = 2
#n_mcruns = 4
n_mcruns_tag = "{:04d}".format(n_mcruns)

# Spectral shift in detector pixels
det_shift_start, det_shift_end, det_shift_increment = -0.3, +1.3, 0.1          # -2.0, +2.0, 0.1
det_shifts = np.arange(det_shift_start, det_shift_end, det_shift_increment)
n_shifts = len(det_shifts)

if reanalyse:
    print("Analysing repeatability impact for dataset {:s}".format(dataset))

    first_config = True
    for config_number in range(0, n_configs):
        config_tag = "{:02d}".format(config_number)

        block_shape = n_shifts, n_mcruns
        xcen_block, fwhm_block = np.zeros(block_shape), np.zeros(block_shape)

        zemax_folder = dataset + '_current_tol_' + config_tag
        # For first dataset, read in parameters from text file and generate IPC and IPG kernels.
        zemax_configuration = zemaxio.read_param_file(dataset, zemax_folder)
        _, wave, _, _, order, im_pix_size = zemax_configuration
        im_oversampling = int(Detector.det_pix_size / im_pix_size)

        if first_config:
            if add_ipc:
                ipc = Ipc(ipc_factor, im_oversampling)
                folder_tag += ipc.get_tag(ipc_factor)
            if add_ipg:
                ipg = Ipg(im_oversampling, 14, 0.44, 14, 0.44)  # Gain profiles, same in x and y
                folder_tag += '_ipg'
            folder_tag += detector.get_tag(Detector.det_pix_size)

            if plot_images:
                if add_ipc:
                    plot_title = zemax_folder + ' Kernel'
                    params = plot_title, None
                    im_kernel, im_oversampling, det_kernel_size = Ipc.kernel
                    title = "Diffusion + IPC ({:6.3f} %)".format(ipc_factor)
                    plot.images([(im_kernel, params)], nrowcol=(1, 1), title=title)
                if add_ipg:
                    ipg_image, _, _, _, _, _ = Ipg.kernel
                    params = 'Intra-pixel gain kernel', 1.0
                    plot.images([(ipg_image, params)], nrowcol=(1, 1))
            first_config = False

        # Calculate dispersion and centre wavelength for each order.
        transform = util.get_polyfit_transform(poly, ech_bounds, zemax_configuration)
        dw_lmspix = util.find_dispersion(transform, zemax_configuration)
        # Select file to analyse
        for mcrun in range(0, n_mcruns):
            mcrun_tag = "{:04d}".format(mcrun)
            fmt = "\r- Config {:02d} of {:d}, Instance {:s} of {:d}"
            print(fmt.format(config_number, n_configs, mcrun_tag, n_mcruns), end="", flush=True)

            filter_tags = [mcrun_tag, 'fits']
            path, file_list = zemaxio.read_file_list(dataset, zemax_folder, filter_tags=filter_tags)
            file_name = file_list[0]
            col_perfect, col_design = 0, 1
            file_id = file_name[0:-5].split('_')[4]
            obs_1 = zemaxio.load_observation(path, file_name, '')

            for res_row, det_shift in enumerate(det_shifts):
                im_shift = det_shift * im_oversampling
                obs_2 = shift.sub_pixel_shift(obs_1, 'spectral', im_shift, debug=False)
                obs_3 = ipc.convolve(obs_2) if add_ipc else obs_2
                title = zemax_folder
                obs_4 = ipg.imprint(obs_3) if add_ipg else obs_3                # Imprint IPG
                obs_5 = detector.measure(obs_4)

                if plot_images:
                    post_label = "Shifted by {:4.1f} pix.".format(det_shift)
                    plot.images([obs_1, obs_3, obs_2, obs_5], nrowcol=(2, 2), plotregion='centre',
                                title=title, pane_titles=['Input', 'Diffusion+IPC', post_label, 'Detector sampled'])

                img_5, par_5 = obs_5
                ndet_rows, ndet_cols = img_5.shape
                row_half_aperture = 2
                row_lo = int(ndet_rows / 2) - row_half_aperture
                row_hi = row_lo + 2 * row_half_aperture + 1
                fit, covar = analyse.fit_gaussian(img_5, row_lo, row_hi, debug=False)
                amp, fwhm, xcen = fit
#                print("det_shift={:10.3f}, xcen={:10.3f}, amp={:10.3f}, fwhm={:10.3f}".format(det_shift, xcen, amp, fwhm))
                xcen_block[res_row, mcrun] = xcen
                fwhm_block[res_row, mcrun] = fwhm
                det_shift += det_shift_increment

        data_id = dataset, folder_tag, config_tag, n_mcruns_tag, axis, ndet_cols
        filer.write_centroids(data_id, det_shifts, xcen_block)

# Read in dataset centroids for plotting.
stats_list = []
for folder_tag in ['_pix_18_00', '_ipc_01_3_pix_18_00']:
    data_list = []
    for config_number in range(0, n_configs):
        config_tag = "{:02d}".format(config_number)
        zemax_folder = dataset + '_current_tol_' + config_tag
        zemax_configuration = zemaxio.read_param_file(dataset, zemax_folder)
        data_id, centroids = filer.read_centroids(dataset, folder_tag, config_tag)
        phases, phase_diffs = analyse.find_phases(data_id, centroids)
        data = data_id, zemax_configuration, phases, phase_diffs
        data_list.append(data)
    stats = analyse.find_stats(data_list)
    stats_list.append((folder_tag, stats))

if plot_profiles:
    plot.stats(stats_list)
    det_pix_pitch = Detector.det_pix_size
    for data in data_list:
        plot.centroids(data, det_pix_pitch)

print('LMS Repeatability (lmsrep.py) - done')
