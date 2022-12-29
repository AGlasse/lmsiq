from lmsiq_analyse import Analyse
from lmsiq_filer import Filer
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
ipg_tag = 'ipg_on' if add_ipg else 'ipg_off'
print("Intra Pixel Gain modelling included = {:s}".format(str(add_ipg)))

plot_images, plot_profiles = False, False
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

n_configs, n_runs = nconfigs_dict[dataset]
#n_configs = 2
#n_runs = 4

# Spectral shift in detector pixels
det_shift_start, det_shift_end, det_shift_increment = -2.0, +2.0, 0.1
det_shifts = np.arange(det_shift_start, det_shift_end, det_shift_increment)
n_shifts = len(det_shifts)

if reanalyse:
    print("Analysing repeatability impact for dataset {:s}".format(dataset))

    for config_number in range(0, n_configs):

        block_shape = n_shifts, n_runs
        xcen_block, fwhm_block = np.zeros(block_shape), np.zeros(block_shape)

        config_tag = "{:02d}".format(config_number)
        zemax_folder = dataset + '_current_tol_' + config_tag
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
        for run_number in range(0, n_runs):
            run_tag = "{:04d}".format(run_number)

            fmt = "\r- Config {:02d} of {:d}, Instance {:s} of {:d}"
            print(fmt.format(config_number, n_configs, run_tag, n_runs), end="", flush=True)

            filter_tags = [run_tag, 'fits']      # Use the first M-C instance in each folder
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

            for res_row, det_shift in enumerate(det_shifts):

                im_shift = det_shift * im_oversampling
                obs_3 = shift.sub_pixel_shift(obs_2, 'spectral', im_shift)
                title = zemax_folder
                obs_ratio = 100 * obs_3[0] / obs_2[0], obs_2[1]
                post_label = "Post-shift by {:4.1f} pix.".format(det_shift)
                if add_ipg:
                    obs_4 = ipg.imprint(obs_3)      # Imprint IPG and detect (rebin to detector pixel scale).
                else:
                    obs_4 = obs_3
                obs_5 = detector.measure(obs_4)

                if plot_images:
                    plot.images([obs_2, obs_3], nrowcol=(2, 1), plotregion='centre',
                                title=title, pane_titles=['Pre-Shift', post_label])
                    plot.images([obs_ratio], nrowcol=(1, 1), plotregion='centre',
                                title=title, pane_titles=[post_label])
                    plot.images([obs_2, obs_3, obs_4, obs_5], nrowcol=(2, 2), plotregion='all',
                                title=title, pane_titles=['Pre-Shift', post_label, 'Post-IPG', 'Frame'])

                img_5, par_5 = obs_5
                nr5, nc5 = img_5.shape
                row_half_aperture = 2
                row_lo = int(nc5 / 2) - row_half_aperture
                row_hi = row_lo + 2 * row_half_aperture + 1
                fit, covar = analyse.fit_gaussian(img_5, row_lo, row_hi, debug=False)
                amp, fwhm, xcen = fit

                xcen_block[res_row, run_number] = xcen
                fwhm_block[res_row, run_number] = fwhm

                det_shift += det_shift_increment
        data_id = dataset, config_tag, ipg_tag, axis

        filer.write_centroids(data_id, det_shifts, xcen_block)

# Read in dataset centroids for plotting.
stats_list = []
for ipg_tag in ['ipg_off', 'ipg_on']:
    data_list = []
    for config_number in range(0, n_configs):
        config_tag = "{:02d}".format(config_number)
        folder = dataset + '_current_tol_' + config_tag
        configuration = zemaxio.read_param_file(dataset, folder)
        data_id = dataset, config_tag, ipg_tag, axis
        centroids = filer.read_centroids(data_id, n_runs)
        centroids = analyse.fix_offset(centroids)
        data = data_id, configuration, centroids
        data_list.append(data)
    stats = analyse.find_stats(data_list)
    stats_list.append((ipg_tag, stats))

if plot_profiles:
    plot.stats(stats_list)

for data in data_list:
    plot.centroids(data)

print('LMS Repeatability (lmsrep.py) - done')
