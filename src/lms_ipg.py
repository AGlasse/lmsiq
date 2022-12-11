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

    def __init__(self, xs, xh, ys, yh):
        """ Generate an Intra Pixel Gain kernel
        """
        gain_kernel_size = 11
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
    def add_ipg(obs):
        """ Convolve the IPG kernel with an image (im1). Returned as im2.  A bit clumsy, but what do you expect?
        """
        im1, params = obs

        im_kernel, oversampling, det_kernel_size = Ipg.kernel
        nr, nc = im1.shape
        nrk, nck = im_kernel.shape
        im2 = np.zeros(im1.shape)
        rc_half = int(det_kernel_size / 2.0)
        for r in range(0, nr-1, oversampling):
            kr1 = 0 if r > oversampling else oversampling - r
            imr1 = r - rc_half * oversampling + kr1
            kr2 = nrk if r < nr - oversampling else nrk - (nr - r)
            imr2 = imr1 + (rc_half + 2) * oversampling - (nrk - kr2) - kr1
            for c in range(0, nc-1, oversampling):
                kc1 = 0 if c > oversampling else oversampling - c
                imc1 = c - rc_half * oversampling + kc1
                kc2 = nck if c < nc - oversampling else nck - (nc - c)
                imc2 = imc1 + (rc_half + 2) * oversampling - (nck - kc2) - kc1
                im1_sub = im1[imr1:imr2, imc1:imc2]
                im_k = im_kernel[kr1:kr2, kc1:kc2]
                im2_sub = im1_sub * im_k
                im2_pix = np.zeros((oversampling, oversampling))
                nrr, ncc = im2_sub.shape
                for rp in range(0, nrr, oversampling):
                    for cp in range(0, ncc, oversampling):
                        im2_pix += im2_sub[rp:rp + oversampling, cp:cp + oversampling]
                im2[r:r+oversampling, c:c+oversampling] += im2_pix
#        print("{:8.3e} -> {:8.3e}".format(im1[0, 0], im2[0, 0]))
        return im2, params
