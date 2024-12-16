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
    det_size = 2048             # H2RG format
    mosaic_format = 2, 2
    mosaic_gap = 2.23           # Gap in mm
    detector_edge_mm = 0.001 * det_size * det_pix_size
    mosaic_edge_mm = detector_edge_mm * mosaic_format[0] + mosaic_gap
    qe = 0.7                                    # QE (el/photon)
    idark = 0.05        # Dark current approx, from Roy. (Finger quotes 0.01)
    rnoise = 70.        # Very approx read noise (Roy model) (Finger/Rauscher use 10 el.)
    q_well = 1.E+5      # Well depth (el.)

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
        det_edge = Detector.nom_pix_pitch * Detector.det_size / 1000.
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
    def down_sample(image_in, im_pix_pitch):
        """ Measure an observation by rebinning at the detector pixel resolution.
        """
        sampling = int(Detector.det_pix_size / im_pix_pitch)        # Image pixels per detector pixel
        nr, nc = image_in.shape
        n_frame_rows, n_frame_cols = int(nr/sampling), int(nc/sampling)
        image_out = image_in.reshape(n_frame_rows, sampling, n_frame_cols, -1).mean(axis=3).mean(axis=1)
        return image_out

    @staticmethod
    def detect(frame, dit, ndit):
        """ Make the mosaic of detector images.
        :return: mosaic - dictionary containing 2x2 mosaic of images
                        'images': synthetic realistic detector images, (PSF convolved and noise added)
                        'cartoons': spatially unresolved image components (illumination map)
                        'reads': gaussian distributed read noise values

        """
        t_int = dit * ndit
        det_shape = Detector.det_size, Detector.det_size
        dark = np.full(det_shape, Detector.idark)
        frame += dark
        image = frame * t_int             # Convert from photocurrent to quantised charge

        rng = np.random.default_rng()
        shot = rng.poisson(np.sqrt(image), det_shape)
        image += shot
        read = rng.normal(loc=0., scale=Detector.rnoise, size=det_shape)
        image += read
        return image

    @staticmethod
    def get_tag(det_pix_size):
        num = int(det_pix_size)
        dec = int(100. * (det_pix_size - num) + 0.5)
        tag = "_pix_{:02d}_{:02d}".format(num, dec)
        return tag
