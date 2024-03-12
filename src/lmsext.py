from lmsiq_analyse import Analyse
from lmsiq_fitsio import FitsIo
from lmsiq_plot import Plot
from lmsiq_phase import Phase
from lms_util import Util
from lms_wcal import Wcal
from lms_ipc import Ipc
from lms_detector import Detector
import numpy as np

print('LMS extended target response (lmsext.py) - started')
print()

detector = Detector()
boo_fmt = "{:40s}= {:s}"

zemaxio = FitsIo()
analyse = Analyse()
plot = Plot()
phase = Phase()
util = Util()
wcal = Wcal()

wcal.read_poly()
inter_pixel = True         # True = Add Inter Pixel Capacitance crosstalk (1.3 % - Rauscher ref.)

plot_images, plot_profiles = True, True
print(boo_fmt.format('Plotting images', str(plot_images)))
print(boo_fmt.format('Plotting profiles', str(plot_profiles)))

run_test = False         # Generate test case with 50 % IPC and an image with one bright pixel.
print(boo_fmt.format('Running test case', str(run_test)))

ifu_slice = 1
location = 'middle'
axes = ['spectral', 'spatial']
axis = 'spectral'
poly_file = '../output/lms_dist_poly_old.txt'           # Distortion polynomial

poly, ech_bounds = util.read_polyfits_file(poly_file)   # Use distortion map for finding dispersion
d_tel = 39.0E9          # ELT diameter in microns

# Spectral shift in detector pixels
det_shift_start, det_shift_end, det_shift_increment = -4.0, +4.0, 0.1          # -2.0, +2.0, 0.1
det_shifts = np.arange(det_shift_start, det_shift_end, det_shift_increment)
n_shifts = len(det_shifts)

print("Analysing repeatability impact on an extended target")
config_number = 99
config_tag = "{:02d}".format(config_number)

im_pix_size = 4.5 / 4.0
im_oversampling = int(Detector.det_pix_size / im_pix_size)

folder_tag = ''
ipc_factor_nominal = 0.013
ipc_factor = 0.013

print("Inter Pixel Capacitance modelling included = {:s}".format(str(inter_pixel)))
ipc = None
if inter_pixel:
    fmt = " - using IPC factor = {:10.3f} (Note nominal value is {:10.3f}"
    print(fmt.format(ipc_factor, ipc_factor_nominal))
    ipc = Ipc(ipc_factor, im_oversampling)
    folder_tag += ipc.get_tag(ipc_factor)
folder_tag += detector.get_tag(Detector.det_pix_size)

if plot_images:
    if inter_pixel:
        im_kernel = Ipc.kernel
        title = "Diffusion + IPC ({:6.3f} %)".format(ipc_factor)
        params = title, None
        plot.images([(im_kernel, params)], nrowcol=(1, 1), title=title)

# Select file to analyse
n_configs = 1
n_mcruns = 1
mcrun = 1
mcrun_tag = "{:04d}".format(mcrun)
fmt = "\r- Config {:02d} of {:d}, Instance {:s} of {:d}"
print(fmt.format(config_number, n_configs, mcrun_tag, n_mcruns), end="", flush=True)

filter_tags = [mcrun_tag, 'fits']
obs_1 = zemaxio.make_extended()
det_shift = 0.
im_shift = det_shift * im_oversampling
obs_2 = phase.sub_pixel_shift(obs_1, 'spectral', im_shift, debug=False)
obs_3 = ipc.convolve(obs_2) if inter_pixel else obs_2
obs_4 = detector.measure(obs_3)

img_3, _ = obs_3
nr, nc = img_3.shape
r_mid = int(nr / 2.)
rhw = int(0.9 * r_mid)
r1, r2 = r_mid - rhw, r_mid + rhw
y = np.sum(img_3[r1:r2, :], axis=1)
x = np.arange(r1, r2)
x = x / im_oversampling
title = "Along row profile"
axs = plot.set_plot_area(title)
ax = axs[0, 0]
ax.set_xlim([2, 10])
ax.set_ylim([0.0, 2.2])
ax.plot_focal_planes(x, y)
plot.show()

print('LMS Extended source (lmsext.py) - done')
