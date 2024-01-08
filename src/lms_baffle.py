#!/usr/bin/env python
"""

"""
import matplotlib.pyplot as plt
import numpy as np
from photutils.aperture import RectangularAperture
from astropy.io import fits
from lms_filer import Filer
from lms_dist_plot import Plot
from lmsiq_analyse import Analyse

print('lms_baffle - Starting')
print('- Program to analyse coronagraph image for baffle location optimisation ')
filer = Filer()
plot = Plot()
analyse = Analyse()

# File locations and names
# output_folder = "../output/distortion/{:s}".format(optical_configuration)
# output_folder = Filer.get_folder(output_folder)
# zem_folder = "../data/distortion/{:s}".format(optical_configuration)
# zem_folder = Filer.get_folder(zem_folder)

baffle_image_file = '../data/baffle/APP_METIS_PSF.fits'
print("- baffle image  = {:s}".format(baffle_image_file))

hdu_list = fits.open(baffle_image_file, mode='readonly')
image, header = hdu_list[0].data, hdu_list[0].header
nrc, _ = image.shape
scale = 0.0013      # 1.6 / nrc
xy_range = nrc * scale
xy1, xy2 = -xy_range / 2., xy_range / 2.

# Define IFU field of view
ifu_x_fov, ifu_y_fov = 0.577, 0.897
ifu_x_off = 0.055
xl, xr = ifu_x_off, ifu_x_off + ifu_x_fov
yb, yt = -ifu_y_fov / 2., ifu_y_fov / 2.
ifu_x = [xl, xl, xr, xr, xl]
ifu_y = [yb, yt, yt, yb, yb]

# Define gap between IFU field of view and beam dump
gap_xy = 1.

# Define beam dump as non-integer rectangle at a non-integer position
dump_attenuation = 1.E-7
dump_c_cov, dump_r_cov = 300, 1000      # Column and row coverage of beam dump
dump = np.full((dump_r_cov, dump_c_cov), dump_attenuation)
drb = int((nrc - dump_r_cov) / 2)

# for dcl in range(0, 1200):      # Iterate over column of left edge of beam dump
#     dcr = dcl + dump_c_cov
#
#     im = np.array(image)        # Copy image
#     im[drb:drt, dcl:dcr] *= dump


xcen, ycen = 0., 0.
ap_pos, ap_width, ap_height = (xcen, ycen), 16., 16.
rect_aper = RectangularAperture(ap_pos, w=ap_width, h=ap_height)
phot = Analyse.exact_rectangular(image, rect_aper)

ax_list = Plot.set_plot_area('FP1 APP PSF')
ax = ax_list[0, 0]
# ax.imshow(image)
# ax.set_xlim([xy1, xy2])
# ax.set_ylim([xy1, xy2])
log_image = np.log10(image)
vpmin, vpmax = -10, 0.

im_map = ax.imshow(log_image, extent=(xy1 - 0.005, xy2 + 0.005, xy1 - 0.005, xy2 + 0.005),
                   interpolation='nearest', cmap='hot', vmin=vpmin, vmax=vpmax, origin='lower')
ax.set_xlabel('arcsec')
ax.set_ylabel('arcsec')
plt.colorbar(mappable=im_map, ax=ax_list, label='log10(Tx)', shrink=0.75)

ax.plot(ifu_x, ifu_y, color='green')
Plot.show()



print('lms_baffle - Done')
