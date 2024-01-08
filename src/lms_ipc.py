#!/usr/bin/env python
"""
@author: Alistair Glasse

Update:
"""
import math
import numpy as np
from lms_globals import Globals
from lmsiq_plot import Plot


class Ipc:

    kernel, ipc_factor = None, None
    tag, folder = None, None
    oversampling, det_kernel_size = None, None
    ipc_factor_nominal, ipc_factor = 0.013, 0.0

    def __init__(self, **kwargs):
        """ Generate a 3 x 3 detector pixel kernel image, which includes diffusion (according to Hardy et al. 2014)
        and inter-pixel capacitance (nominal 1.3 % - Rauscher ref.)
        """
        loss = kwargs.get('loss', 0.5)

        det_kernel_size = 3
        oversampling = Globals.get_im_oversampling('raw_zemax')
        kernel_size = det_kernel_size * oversampling
        kernel_shape = kernel_size, kernel_size
        kernel = np.zeros(kernel_shape)
        # Generate diffusion component  A exp(-B x^2) where A = 1 and B = 2.4 to minimise transmission losses
        kernel = Ipc.add_diffusion(kernel, oversampling)
        kernel = Ipc.add_ipc(kernel, oversampling, det_kernel_size)
        # Finally, re-normalise the kernel to unity (no signal lost from image)
        renorm = np.sum(kernel) * (1. + loss)
        kernel /= renorm
        Ipc.kernel = kernel
        Ipc.oversampling, Ipc.det_kernel_size = oversampling, det_kernel_size
        Ipc.plot_kernel()
        return

    @staticmethod
    def test():
        params = 'file_id', 1.
        image1 = np.zeros((128, 128)) + 1.0      # make an extended image
        obs1 = image1, params
        obs_con1 = Ipc.convolve(obs1)
        image2 = np.zeros((128, 128))
        image2[64, 64] = 1.0
        obs2 = image2, params
        obs_con2 = Ipc.convolve(obs2)
        sum2 = np.sum(obs_con2[0])
        image3 = np.zeros((128, 128))
        image3[66, 66] = 1.0
        obs3 = image3, params
        obs_con3 = Ipc.convolve(obs3)
        sum3 = np.sum(obs_con3[0])

        Plot.images([obs1, obs2, obs3, obs_con1, obs_con2, obs_con3], nrowcol=(2, 3), title='Ipc Test',
                    shrink=0.1,
                    png_path='../output/convolution_test')

        return

    @staticmethod
    def set_inter_pixel(inter_pixel):
        Ipc.ipc_factor = Ipc.ipc_factor_nominal if inter_pixel else 0.0
        Ipc.tag = 'ipc_on' if inter_pixel else 'ipc_off'
        Ipc.folder = Ipc.tag + '/'
        return

    @staticmethod
    def plot_kernel():
        title = "Diffusion + IPC ({:6.3f} %)".format(Ipc.ipc_factor)
        params = title, None

        Plot.images([(Ipc.kernel, params)],
                    nrowcol=(1, 1), title=title, png_path='../output/kernel')
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
    def add_ipc(kernel, oversampling, det_kernel_size):
        """ Add in IPC
        :param kernel:
        :param oversampling:
        :param det_kernel_size:
        :return:
        """
        rc1, rc2 = oversampling, 2 * oversampling
        sig_cen = np.sum(kernel[rc1:rc2, rc1:rc2])
        ovs2 = oversampling * oversampling          # Normalise IPC to image pixels
        ipc_sig = Ipc.ipc_factor * sig_cen / ovs2
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
        kernel = Ipc.kernel
        max_val = kwargs.get('max_val', None)
        # Plot column combined profile for comparison with Hardy et al.
        im_k = np.sum(kernel, axis=0)
        if max_val is not None:
            f = max_val / np.amax(im_k)
            im_k *= f
        im_x = np.arange(0, len(im_k)) / Ipc.oversampling
        axs = Plot.set_plot_area(title=title)
        ax = axs[0, 0]
        ax.plot_focal_planes(im_x, im_k, ls='none', marker='o', ms=5.0, color='black')
        for x in range(0, Ipc.det_kernel_size):
            ax.plot_focal_planes([x, x], [0., np.amax(im_k)], lw=0.5, color='grey')
        Plot.show()
        return

    @staticmethod
    def convolve(obs):
        """ Convolve the IPC kernel with an image (im1). Returned as im2.  A bit clumsy.
        """
        im1, params = obs
        kernel, oversampling, det_kernel_size = Ipc.kernel, Ipc.oversampling, Ipc.det_kernel_size
        knorm = np.sum(kernel) * oversampling * oversampling
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
        im2 *= knorm
#        print("Ipc.convolve, {:8.3e} -> {:8.3e}".format(np.sum(im1), np.sum(im2)))
        return im2, params
