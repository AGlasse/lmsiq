#!/usr/bin/env python
"""
Inter pixel capacitance object.

@author: Alistair Glasse

Update:
"""
import numpy as np
from lms_globals import Globals


class Detector:

    nom_pix_pitch = Globals.nom_pix_pitch
    det_pix_size = nom_pix_pitch        # / 4.0

    def __init__(self):
        """ Detector object, mainly used to sample/measure Zemax observations
        """
        fmt = "Detector pixel size = {:10.3f} micron (Note 'true' LMS det pixel size = {:10.3f}"
        print(fmt.format(Detector.det_pix_size, Detector.nom_pix_pitch))
        return

    @staticmethod
    def measure(observation):
        """ Measure an observation by rebinning at the detector pixel resolution.
        """
        image_in, params = observation
        _, im_pix_pitch = params
        sampling = int(0.001 * Detector.det_pix_size / im_pix_pitch)        # Image pixels per detector pixel
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
    def set_flat(observation):
        image, params = observation
        image = np.full(image.shape, 1.0)
        return image, params

    @staticmethod
    def get_tag(det_pix_size):
        num = int(det_pix_size)
        dec = int(100. * (det_pix_size - num) + 0.5)
        tag = "_pix_{:02d}_{:02d}".format(num, dec)
        return tag
