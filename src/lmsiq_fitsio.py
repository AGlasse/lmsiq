from os.path import join
import shutil
import numpy as np
from astropy.io import fits


class FitsIo:

    data_path, zip_path = '', ''

    def __init__(self, optical_path):
        FitsIo.data_path = '../data/iq/' + optical_path + '/'
        FitsIo.zip_path = '../zip/'
        return

    @staticmethod
    def _copy(path, folder):
        shutil.copy(path, folder)
        print("Copying {:s} to {:s}".format(path, folder))
        return

    @staticmethod
    def read_zemax(path, f):
        file_path = join(path, f)
        hdu_list = fits.open(file_path, mode='readonly')
        hdu = hdu_list[0]
        return hdu.data, hdu.header

    @staticmethod
    def make_extended():
        image = np.full((512, 512), 1.0)
        params = 'Extended', 0.018 / 16.
        return image, params

    @staticmethod
    def write_cube(cube, cube_name, iq_filer):
        fits_folder = iq_filer.get_folder(iq_filer.output_folder + 'cube/fits')
        fits_path = fits_folder + cube_name
        hdu = fits.PrimaryHDU(cube)
        hdu.writeto(fits_path + '.fits', overwrite=True)
        return

    @staticmethod
    def read_cube(process_level, ipc_tag, iq_filer):
        file_path = iq_filer.cube_path + process_level + '_' + ipc_tag + '_cube.fits'
        hdul = fits.open(file_path, readonly=True)
        hdu = hdul[0]
        data = hdu.data
        n_waves, _, _, _ = data.shape
        waves = []
        alpha_oversampling = hdu.header['ALPHA_OS']
        for i in range(n_waves):
            kw = "WAV{:d}".format(i)
            wave = hdu.header[kw]
            waves.append(wave)
        return data, waves, alpha_oversampling
