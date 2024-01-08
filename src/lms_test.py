#!/usr/bin/env python
"""

"""
from lms_dist_util import Util
from lms_wcal import Wcal
from lms_globals import Globals
from lms_efficiency import Efficiency
from lms_dist_plot import Plot

print('lms_test - Starting')
util = Util()
plot = Plot()
poly_file = '../output/lms_dist_poly_old.txt'

# Define test point(s) (wavelength/um, alpha/as, beta/as)
wave = 4.70
alpha = 0.3
fp2_x = alpha / Globals.efp_as_mm
beta = 0.1
slice = int(15 + 28 * beta / Globals.beta_fov)
print("Slice= {:d}".format(slice))

# Find configuration for this wavelength
wcal = Wcal()
Wcal.read_poly()
ech_order, ech_angle = wcal.find_echelle_setting(wave)
print("Echelle order= {:d}, Echelle angle = {:10.3f}".format(ech_order, ech_angle))

blaze = Efficiency()
eff_blaze = Efficiency.blaze(wave, ech_order)
eff_slow = Efficiency.slow(wave)
eff = Efficiency.combined(wave, ech_order)
print('At this setting..')
print('Blaze efficiency =    {:10.3f}'.format(eff_blaze))
print('Slow efficiency =     {:10.3f}'.format(eff_slow))
print('Combined efficiency = {:10.3f}'.format(eff))

w_test_lim = [3.0, 5.0]
weoas = Efficiency.test(w_test_lim)                         # Generate efficiency curves for adjacent orders
plot.efficiency_v_wavelength(weoas, xlim=w_test_lim)        # Plot blaze profiles (cf SPIE 11451_49)

phase = wave * ech_order / (2.0 * Globals.rule_spacing)
print("Phase= {:10.3f}, FP2_X = {:10.3f}".format(phase, fp2_x))

poly, ech_bounds = util.read_polyfits_file(poly_file)
transform = util.get_polyfit_transform(poly, ech_bounds, ech_order, slice, ech_angle)
util.print_poly_transform(transform)

a, b, ai, bi = transform
det_x, det_y = util.apply_distortion([phase], [fp2_x], a, b)
print("Det_x= {:10.3f}, Det_y= {:10.3f}".format(det_x[0], det_y[0]))
pix_x = det_x[0] / Globals.mm_pix
pix_y = det_y[0] / Globals.mm_pix

print("Pix_x= {:10.3f}, Pix_y= {:10.3f}".format(pix_x, pix_y))
print("lms_test - Done.")
