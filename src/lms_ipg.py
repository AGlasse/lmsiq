#!/usr/bin/env python
"""
Inter pixel capacitance object.

@author: Alistair Glasse

Update:
"""
import numpy as np
import math


class Ipg:

    factor = 0.0
    kernel = None

    def __init__(self, im_oversampling, xs, xh, ys, yh):
        """ Generate an Intra Pixel Gain kernel which samples the detector pixel with 4 x the image sampling or TBD..
        """
        gain_kernel_size = 4 * im_oversampling
        gain_kernel = np.zeros((gain_kernel_size, gain_kernel_size))
        dxy = 1.0 / (gain_kernel_size - 1)
        x_coords = np.arange(-0.5, +0.5 + dxy, dxy)
        y_coords = np.arange(-0.5, +0.5 + dxy, dxy)
        for c in range(0, gain_kernel_size):
            x = x_coords[c]
            gx = Ipg._gain_profile(x, xs, xh)
            for r in range(0, gain_kernel_size):
                y = y_coords[r]
                gy = Ipg._gain_profile(y, ys, yh)
                gain_kernel[r, c] = gx * gy
        Ipg.kernel = gain_kernel, gain_kernel_size, xs, xh, ys, yh
        return

    @staticmethod
    def _gain_profile(u, s, h):
        """ Calculate Fermi-like profile.
        -0.5 < u < +0.5 - pixel coordinate
        c - gain profile edge steepness (~10 - 20 looks ok)
        h - value of u for which gain is 0.5
        """
        factor = 0.5 * s * (math.fabs(u) / h - 1.0)
        gain = 1.0 / (math.exp(factor) + 1)
        return gain

    @staticmethod
    def imprint(obs):
        """ Imprint the IPG profile onto all pixels in an image.
        """
        im1, params = obs
        im_oversampling = 4
        ipg_kernel, kernel_size, xs, xh, ys, yh = Ipg.kernel

        nr_im, nc_im = im1.shape
        nr_det, nc_det = int(nr_im / im_oversampling), int(nc_im / im_oversampling)
        ipg_oversampling = 16

        ipg_im = int(ipg_oversampling / im_oversampling)
        im2 = np.zeros(im1.shape)
        # Select pixel equivalent image regions matching a pixel
        for r_det in range(0, nr_det):
            r_im_start = r_det * im_oversampling
            for c_det in range(0, nc_det):
                c_im_start = c_det * im_oversampling
                # Resample image pixel data to match IPG kernel.
                for r_im_off in range(0, im_oversampling):
                    r_im = r_im_start + r_im_off
                    r_ipg_start = r_im_off * ipg_im
                    for c_im_off in range(0, im_oversampling):
                        c_im = c_im_start + c_im_off
                        c_ipg_start = c_im_off * ipg_im
                        im_pix_val = im1[r_im, c_im]
                        r1, c1 = r_ipg_start, c_ipg_start
                        r2, c2 = r1 + ipg_im, c1 + ipg_im
                        im_pix = im_pix_val * ipg_kernel[r1:r2, c1:c2]
#                        fmt = "Imprinting kernel r={:3d}:{:3d}, c={:3d}:{:3d} onto image pixel {:3d}:{:3d}"
#                        print(fmt.format(r1, r2, c1, c2, r_im, c_im))
                        im2[r_im, c_im] = np.mean(im_pix)
        return im2, params
