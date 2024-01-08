import os
from os import listdir
from os.path import join
import shutil
import numpy as np
from astropy.io import fits
from lms_filer import Filer


class FitsIo:

    data_path, zip_path = '', ''

    def __init__(self,optical_path):
        FitsIo.data_path = '../data/iq/' + optical_path + '/'
        FitsIo.zip_path = '../zip/'
        return

    @staticmethod
    def _copy(path, folder):
        shutil.copy(path, folder)
        print("Copying {:s} to {:s}".format(path, folder))
        return

    @staticmethod
    def setup_zemax_configuration(config):
        optical_path, date_stamp, n_wavelengths, n_mcruns, slice_locs, folder_name, config_label = config
        wave_tag = "{:02d}/".format(0)
        data_folder = date_stamp + folder_name + wave_tag
        zemax_configuration = FitsIo.read_param_file(date_stamp, data_folder)
        _, _, _, _, order, im_pix_size = zemax_configuration
        return im_pix_size, zemax_configuration

    @staticmethod
    def copy_from_zip(config):
        dataset, n_wavelengths, n_mcruns, slice_locs, folder_name, config_label = config
        zip_folder_list = listdir(FitsIo.zip_path)
        dataset_folder = FitsIo.data_path + 'psf_model_' + dataset + '_multi_wavelength/'
        if not os.path.exists(dataset_folder):
            os.mkdir(dataset_folder)
        for folder in zip_folder_list:
            data_folder = dataset_folder + folder
            if not os.path.exists(data_folder):
                os.mkdir(data_folder)
            zip_folder = FitsIo.zip_path + folder + '/'
            zip_files = listdir(zip_folder)
            for zip_file in zip_files:
                if '.txt' in zip_file:
                    zip_path = zip_folder + zip_file
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
    def read_file_list(dataset, folder, **kwargs):
        filter_tags = kwargs.get('filter_tags', [])
        parent_folder = FitsIo.data_path + 'psf_model_' + dataset + '_multi_wavelength/'
        path = parent_folder + folder
        file_list = listdir(path)
        for filter_tag in filter_tags:
            file_list = [f for f in file_list if filter_tag in f]
        return path, file_list

    @staticmethod
    def read_param_file(date_stamp, folder):
        parent_folder = FitsIo.data_path + 'psf_model_' + date_stamp + '_multi_wavelength/'
        path = parent_folder + folder + folder[0:-1] + '.txt'
        pf = open(path, 'r')
        lines = pf.read().splitlines()
        pf.close()
        slice_no = int(lines[1].split(':')[1])
        wave = float(lines[3].split(':')[1])
        prism_angle = float(lines[4].split(':')[1])
        grating_angle = float(lines[5].split(':')[1])
        order = -1 * int(lines[6].split(':')[1])
        pix_size = float(lines[24].split(':')[1])
        return slice_no, wave, prism_angle, grating_angle, order, pix_size

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
    def load_dataset(dataset, folder, n_mcruns, **kwargs):
        """ Load image and parameter information as a list of observation objects from a list of files, sorting them
        in order 'perfect, design then obs_id = 00, 01 etc.
        :param n_mcruns:
        :param folder:
        :param dataset:
        :return observations:  List of image, params tuples.
        """
        observations = []
        filter_tags = ['perfect', 'fits']
        path, file_list = FitsIo.read_file_list(dataset, folder, filter_tags=filter_tags)
        if len(file_list) < 1:
            print(folder)
            print(len(file_list))

        file_name = file_list[0]
        obs = FitsIo.load_observation(path, file_name, '')
        observations.append(obs)

        filter_tags = ['design', 'fits']
        path, file_list = FitsIo.read_file_list(dataset, folder, filter_tags=filter_tags)
        file_name = file_list[0]
        obs = FitsIo.load_observation(path, file_name, '')
        observations.append(obs)

        # Select file to analyse
        for mcrun in range(0, n_mcruns):
            mcrun_tag = "{:04d}".format(mcrun)

            filter_tags = [mcrun_tag, 'fits']
            path, file_list = FitsIo.read_file_list(dataset, folder, filter_tags=filter_tags)
            file_name = file_list[0]
            obs = FitsIo.load_observation(path, file_name, '')
            observations.append(obs)
        return observations

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
    def write_cube(process_level, ipc_tag, waves, alpha_oversampling, cube):
        file_path = Filer.cube_path + process_level + '_' + ipc_tag + '_cube.fits'
        hdu = fits.PrimaryHDU(cube)
        hdu.header['ALPHA_OS'] = alpha_oversampling
        for i, wave in enumerate(waves):
            kw = "WAV{:d}".format(i)
            hdu.header[kw] = wave
        hdu.writeto(file_path, overwrite=True)
        return

    @staticmethod
    def read_cube(process_level, ipc_tag):
        file_path = Filer.cube_path + process_level + '_' + ipc_tag + '_cube.fits'
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
