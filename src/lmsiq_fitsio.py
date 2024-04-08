import os
from os import listdir
from os.path import join
import shutil
import numpy as np
from astropy.io import fits
from lms_filer import Filer


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
    def setup_zemax_configuration(data_identifier):

        optical_path, date_stamp, n_wavelengths, n_mcruns, slice_locs, folder_name, config_label = data_identifier
        wave_tag = "{:02d}/".format(0)
        data_folder = date_stamp + folder_name + wave_tag
        zemax_configuration = FitsIo.read_param_file(date_stamp, data_folder)
        _, _, _, _, order, im_pix_size = zemax_configuration
        return im_pix_size, zemax_configuration

    @staticmethod
    def get_data_table(data_identifier):
        optical_configuration, date_stamp, n_wavelengths, n_mcruns, \
            par_file, zim_locs, folder_name, config_label \
            = data_identifier
        data_table = {'file_path': [], 'mc_id': [],       # Monte-Carlo ID (-1 des, -2 perf)
                      'centre_slice': [], 'spifu_slice': [],
                      'field_x': [], 'field_y': [],
                      'im_pixel_size': [],
                      }
        zip_folder = FitsIo.zip_path + date_stamp + '/'
        zip_folder_list = listdir(zip_folder)
        for folder in zip_folder_list:
            zip_sub_folder = zip_folder + folder + '/'
            zip_files = listdir(zip_sub_folder)
            parameters = None
            for file in zip_files:
                if '.txt' in file:
                    par_path = zip_sub_folder + file
                    parameters = FitsIo.read_param_file(par_path)
                    continue
            slice_no, wave, prism_angle, grating_angle, order, pix_size = parameters
            for zip_file in zip_files:
                print(zip_file)
                data_table['file_path'].append(zip_file)
        return data_table

    @staticmethod
    def copy_from_zip(data_identifier):
        optical_configuration, date_stamp, n_wavelengths, n_mcruns, zim_locs, _, folder_name, config_label \
            = data_identifier
        zip_folder = FitsIo.zip_path + date_stamp + '/'
        zip_folder_list = listdir(zip_folder)
        dataset_folder = Filer.get_folder(FitsIo.data_path + date_stamp)
        # dataset_folder = FitsIo.data_path + date_stamp + '/'
        # if not os.path.exists(dataset_folder):
        #     os.mkdir(dataset_folder)
        for folder in zip_folder_list:
            data_folder = dataset_folder + folder
            if not os.path.exists(data_folder):
                os.mkdir(data_folder)
            zip_sub_folder = zip_folder + folder + '/'
            zip_files = listdir(zip_sub_folder)
            for zip_file in zip_files:
                if '.txt' in zip_file:
                    zip_path = zip_sub_folder + zip_file
                    FitsIo._copy(zip_path, data_folder)
                    continue
                tokens = zip_file.split('_')
                slice_token = tokens[4]
                slice_folder = data_folder + '/ifu'     # Initialise to be ifu/slicer data
                if slice_token != 'slicer':
                    slice_no = int(slice_token)
                    slice_folder = data_folder + "/slice_{:d}".format(slice_no)
                if not os.path.exists(slice_folder):
                    os.mkdir(slice_folder)
                zip_path = zip_folder + zip_file
                FitsIo._copy(zip_path, slice_folder)
        return

    @staticmethod
    def read_file_list(date_stamp, folder, **kwargs):
        filter_tags = kwargs.get('filter_tags', [])
        parent_folder = FitsIo.data_path + date_stamp + '/'
        path = parent_folder + folder
        file_list = listdir(path)
        for filter_tag in filter_tags:
            file_list = [f for f in file_list if filter_tag in f]
        return path, file_list

    @staticmethod
    def read_param_file(path):
        pf = open(path, 'r')
        lines = pf.read().splitlines()
        pf.close()
        slice_no = int(lines[1].split(':')[1])
        wave = float(lines[3].split(':')[1])
        prism_angle = float(lines[4].split(':')[1])
        grating_angle = float(lines[5].split(':')[1])
        order = -1 * int(lines[6].split(':')[1])
        pix_size = float(lines[24].split(':')[1])
        parameters = slice_no, wave, prism_angle, grating_angle, order, pix_size
        return parameters

    @staticmethod
    def load_observation(path, file_name, file_id):
        """ Load a single Zemax observation
        :param path:
        :param file_name
        :param file_id
        :return observation  Image, params tuple.
        """
        image, header = FitsIo.read_zemax(path, file_name)

        mm_fitspix = 1000.0 * header['CDELT1']
        params = file_id, mm_fitspix
        return image, params

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
        fits_folder = Filer.get_folder(iq_filer.output_folder + 'cube/fits')
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
