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

    det_pix_size = nom_pix_pitch
    pix_edge = 2048             # H2RG format
    mosaic_format = 2, 2
    mosaic_gap = 2.23           # Gap in mm
    detector_edge_mm = 0.001 * pix_edge * det_pix_size
    mosaic_edge_mm = detector_edge_mm * mosaic_format[0] + mosaic_gap

    def __init__(self):
        """ Detector object, mainly used to sample/measure Zemax observations
        """
        fmt = "Detector pixel size = {:10.3f} micron (Note 'true' LMS det pixel size = {:10.3f}"
        print(fmt.format(Detector.det_pix_size, Detector.nom_pix_pitch))
        return

    @staticmethod
    def get_bounds():
        """ Get the bounds of all four detectors in mm.
        """
        det_edge = Detector.nom_pix_pitch * Detector.pix_edge / 1000.
        det_gap = Detector.mosaic_gap
        xy_lim = det_edge + det_gap / 2.
        xbl = np.array([-xy_lim, -xy_lim, -xy_lim + det_edge, -xy_lim + det_edge])
        ybl = np.array([-xy_lim, -xy_lim + det_edge, -xy_lim, -xy_lim + det_edge])

        x, y = np.zeros(16), np.zeros(16)
        xy_offset = det_edge + det_gap
        x[0:4], y[0:4] = xbl, ybl
        x[4:8], y[4:8] = xbl + xy_offset, ybl
        x[8:12], y[8:12] = xbl, ybl + xy_offset
        x[12:16], y[12:16] = xbl + xy_offset, ybl + xy_offset
        return x, y

    @staticmethod
    def measure(image_in, im_pix_pitch):
        """ Measure an observation by rebinning at the detector pixel resolution.
        """
        sampling = int(Detector.det_pix_size / im_pix_pitch)        # Image pixels per detector pixel
        nr, nc = image_in.shape
        n_frame_rows, n_frame_cols = int(nr/sampling), int(nc/sampling)
        image_out = np.zeros((n_frame_rows, n_frame_cols))
        for r in range(0, n_frame_rows):
            r1 = r * sampling
            r2 = r1 + sampling
            for c in range(0, n_frame_cols):
                c1 = c * sampling
                c2 = c1 + sampling
                image_out[r, c] = np.mean(image_in[r1:r2, c1:c2])
        return image_out





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
