#!/usr/bin/env python
"""
Inter pixel capacitance object.

@author: Alistair Glasse

Update:
"""
import numpy as np
import math


class Detector:


    def __init__(self):
        """ Detector object, mainly used to sample/measure Zemax observations
        """
        return

    @staticmethod
    def measure(observation):
        """ Measure an observation by rebinning onto the pixel scale
        """
        image_in, params = observation
        sampling = 4            # Pixels per detector pixel
        nr, nc = image_in.shape
        n_frame_rows, n_frame_cols = int(nr/sampling), int(nc/sampling)
        frame = np.zeros((n_frame_rows, n_frame_cols))
        for r in range(0, n_frame_rows):
            r1 = r * sampling
            r2 = r1 + sampling
            for c in range(0, n_frame_cols):
                c1 = c * sampling
                c2 = c1 + sampling
                frame[r, c] = np.mean(image_in[r1:r2, c1:c2])

        return frame, params

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
