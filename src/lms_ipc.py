#!/usr/bin/env python
"""
@author: Alistair Glasse

Update:
"""
import math
import numpy as np
from lmsiq_plot import LMSIQPlot as Plot


class Ipc:

    kernel, ipc_factor = None, None
    oversampling, det_kernel_size = None, None

    def __init__(self, ipc_factor, oversampling):
        """ Generate a 3 x 3 detector pixel kernel image, which includes diffusion (according to Hardy et al. 2014)
        and inter-pixel capacitance (nominally 1.6 % in detector pixel space)
        """
        det_kernel_size = 3
        kernel_size = det_kernel_size * oversampling
        kernel_shape = kernel_size, kernel_size
        kernel = np.zeros(kernel_shape)
        # Generate diffusion component  A exp(-B x^2) where A = 1 and B = 2.4 to minimise transmission losses
        kernel = Ipc.add_diffusion(kernel, oversampling)
        kernel = Ipc.add_ipc(kernel, oversampling, ipc_factor, det_kernel_size)
        # Finally, re-normalise the kernel to unity (no signal lost from image)
        renorm = np.sum(kernel)
        kernel /= renorm
        Ipc.kernel, Ipc.ipc_factor = kernel, ipc_factor
        Ipc.oversampling, Ipc.det_kernel_size = oversampling, det_kernel_size
        return

    @staticmethod
    def add_diffusion(im_kernel, oversampling):
        nrows, ncols = im_kernel.shape
        a, b = 1.0, 0.65
        bsq = b * b
        rc, cc = (nrows / 2.0) - 0.5, (ncols / 2.0) - 0.5
        for r in range(0, nrows):
            x = (r - rc) / oversampling
            for c in range(0, ncols):
                y = (c - cc) / oversampling
                d = math.sqrt(x*x + y*y)
                diff = a * math.exp(-d * d / bsq)
                im_kernel[r, c] = diff
        return im_kernel

    @staticmethod
    def add_ipc(kernel, oversampling, ipc_factor, det_kernel_size):

        # Add in IPC
        rc1, rc2 = oversampling, 2 * oversampling
        sig_cen = np.sum(kernel[rc1:rc2, rc1:rc2])
        ipc_sig = ipc_factor * sig_cen
        col_profile = np.sum(kernel, axis=0)
        det_kernel_shape = det_kernel_size, det_kernel_size
        det_kernel = np.zeros(det_kernel_shape)
        det_kernel[0, 1], det_kernel[1, 0], det_kernel[1, 2], det_kernel[2, 1] = ipc_sig, ipc_sig, ipc_sig, ipc_sig

        for r in range(0, det_kernel_size):
            imr1 = r * oversampling
            imr2 = imr1 + oversampling
            for c in range(0, det_kernel_size):
                imc1 = c * oversampling
                imc2 = imc1 + oversampling
                kernel[imr1:imr2, imc1:imc2] += det_kernel[r, c]
        return kernel

    @staticmethod
    def plot_kernel_profile(title, **kwargs):
        plot = Plot()
        kernel = Ipc.kernel
        max_val = kwargs.get('max_val', None)
        # Plot column combined profile for comparison with Hardy et al.
        im_k = np.sum(kernel, axis=0)
        if max_val is not None:
            f = max_val / np.amax(im_k)
            im_k *= f
        im_x = np.arange(0, len(im_k)) / Ipc.oversampling
        axs = plot.set_plot_area(title)
        ax = axs[0, 0]
        ax.plot(im_x, im_k, ls='none', marker='o', ms=5.0, color='black')
        for x in range(0, Ipc.det_kernel_size):
            ax.plot([x, x], [0., np.amax(im_k)], lw=0.5, color='grey')
        plot.show()
        return

    @staticmethod
    def get_tag(ipc_factor):
        pc = 100. * ipc_factor
        int_pc = int(pc)
        dec = int(10. * (pc - int_pc) + 0.5)
        tag = "_ipc_{:02d}_{:01d}".format(int_pc, dec)
        return tag

    @staticmethod
    def convolve(obs):
        """ Convolve the IPC kernel with an image (im1). Returned as im2.  A bit clumsy.
        """
        im1, params = obs

        kernel, oversampling, det_kernel_size = Ipc.kernel, Ipc.oversampling, Ipc.det_kernel_size

        nr, nc = im1.shape
        nrk, nck = kernel.shape
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
                im_k = kernel[kr1:kr2, kc1:kc2]
                im2_sub = im1_sub * im_k
                im2_pix = np.zeros((oversampling, oversampling))
                nrr, ncc = im2_sub.shape
                for rp in range(0, nrr, oversampling):
                    for cp in range(0, ncc, oversampling):
                        im2_pix += im2_sub[rp:rp + oversampling, cp:cp + oversampling]
                im2[r:r+oversampling, c:c+oversampling] += im2_pix
#        print("{:8.3e} -> {:8.3e}".format(im1[0, 0], im2[0, 0]))
        return im2, params
