#!/usr/bin/env python
"""
@author: Alistair Glasse

Update:
"""
import numpy as np


class Ipc:

    kernel = None
    factor = 0.0

    def __init__(self, factor, oversampling):
        """ Generate an IPC kernel (nominally 1.6 % in detector pixel space)
        """
        det_kernel_size = 3
        det_kernel_shape = det_kernel_size, det_kernel_size
        det_kernel = np.zeros(det_kernel_shape)
        det_kernel[1, 1] = 1.0
        det_kernel[0, 1], det_kernel[1, 0], det_kernel[1, 2], det_kernel[2, 1] = factor, factor, factor, factor
        im_kernel_size = det_kernel_size * oversampling
        im_kernel_shape = im_kernel_size, im_kernel_size
        im_kernel = np.zeros(im_kernel_shape)
        for r in range(0, det_kernel_size):
            imr1 = r * oversampling
            imr2 = imr1 + im_kernel_size
            for c in range(0, det_kernel_size):
                imc1 = c * oversampling
                imc2 = imc1 + im_kernel_size
                im_kernel[imr1:imr2, imc1:imc2] = det_kernel[r, c]
        Ipc.factor = factor
        Ipc.oversampling = oversampling
        Ipc.kernel = im_kernel, oversampling, det_kernel_size
        return

    @staticmethod
    def convolve(obs):
        """ Convolve the IPC kernel with an image (im1). Returned as im2.  A bit clumsy.
        """
        im1, params = obs

        im_kernel, oversampling, det_kernel_size = Ipc.kernel
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
