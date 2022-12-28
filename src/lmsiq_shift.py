from os import listdir
from os.path import isfile, join
import numpy as np


class LMSIQShift:


    def __init__(self):
        return

    @staticmethod
    def sub_pixel_shift(observation, axis, sp_shift, **kwargs):
        debug = kwargs.get('debug', False)

        image_in, params = observation
        resolution = kwargs.get('resolution', 10)   # Number of sub-pixels per image pixel
        if axis == 'spectral':                      # Rotate input image
            np.moveaxis(image_in, 0, 1)
        nrows, ncols = image_in.shape
        ncols_ss = resolution * ncols
        image_ss = np.zeros((nrows, ncols_ss))        # Create super-sampled image
        col_shift = int(sp_shift * resolution)
        if debug:
            fmt = "Shifting {:s} axis image, ({:d}x) oversampled by {:d} columns"
            print(fmt.format(axis, resolution, col_shift))
        for col in range(0, ncols):     # Map shifted image data into super-sampled image
            c1 = col * resolution + col_shift
            c2 = c1+resolution
            c1 = c1 if c1 > 0 else 0
            c2 = c2 if c2 < ncols_ss else ncols_ss - 1
            for c_ss in range(c1, c2):
                image_ss[:, c_ss] = image_in[:, col]
        image_out = np.zeros(image_in.shape)
        for col in range(0, ncols):     # Resample super-sampled image onto output image
            col_ss = col * resolution
            strip = np.mean(image_ss[:, col_ss:col_ss + resolution], axis=1)
            image_out[:, col] = strip
        if axis == 'spectral':                      # Rotate input image
            np.moveaxis(image_out, 0, 1)
        return image_out, params
