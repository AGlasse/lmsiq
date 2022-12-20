from os import listdir
from os.path import isfile, join
from astropy.io import fits
from lms_globals import Globals

class LMSIQZemaxio:


    def __init__(self):
        LMSIQZemaxio.data_path = '../data/'
        return

    @staticmethod
    def read_file_list(dataset, folder, **kwargs):
        filter_tags = kwargs.get('filter_tags', [])
        parent_folder = LMSIQZemaxio.data_path + 'psf_model_' + dataset + '_multi_wavelength/'
        path = parent_folder + folder
        file_list = listdir(path)
        for filter_tag in filter_tags:
            file_list = [f for f in file_list if filter_tag in f]
        return path, file_list

    @staticmethod
    def read_param_file(rundate, folder):
        parent_folder = LMSIQZemaxio.data_path + 'psf_model_' + rundate + '_multi_wavelength/'
        path = parent_folder + folder + '/' + folder + '.txt'
        pf = open(path, 'r')
        lines = pf.read().splitlines()
        pf.close()
        slice = int(lines[1].split(':')[1])
        wave = float(lines[3].split(':')[1])
        prism_angle = float(lines[4].split(':')[1])
        grating_angle = float(lines[5].split(':')[1])
        order = -1 * int(lines[6].split(':')[1])
        pix_size = float(lines[24].split(':')[1])
        return slice, wave, prism_angle, grating_angle, order, pix_size

    @staticmethod
    def load_observation(path, file_name, file_id):
        """ Load a single Zemax observation
        :param path:
        :param file_name
        :param file_id
        :return observation  Image, params tuple.
        """
        image, header = LMSIQZemaxio.read_zemax_fits(path, file_name)
        mm_fitspix = 1000.0 * header['CDELT1']
        params = file_id, mm_fitspix
        return image, params

    @staticmethod
    def load_dataset(path, file_list, **kwargs):
        """ Load image and parameter information as a list of observation objects from a list of files, sorting them
        in order 'perfect, design then obs_id = 00, 01 etc.
        :param path:
        :param file_list:
        :param run_test: Boolean.  True = Replace image data with a single bright detector pixel at the image centre.
        :return observations:  List of image, params tuples.
        """
        run_test = kwargs.get('run_test', False)

        n_obs = len(file_list)
        observations = [None]*n_obs
        for file in file_list:
            file_id = file[0:-5].split('_')[4]
            if file_id == 'perfect':
                j = 0
            else:
                if file_id == 'design':
                    j = 1
                else:
                    j = int(file_id) + 2
            image, params = LMSIQZemaxio.load_observation(path, file, file_id)
            if run_test:
                mm_fitspix = params[1]
                oversampling = int((Globals.mm_lmspix / mm_fitspix) + 0.5)
                image[:, :] = 0.0
                nr, nc = image.shape
                r1, c1 = int((nr - oversampling) / 2.0), int((nc - oversampling) / 2.0)
                r2, c2 = r1 + oversampling, c1 + oversampling
                image[r1:r2, c1:c2] = 1.0
            observations[j] = image, params
        return observations

    @staticmethod
    def read_zemax_fits(path, f):
        file_path = join(path, f)
        hdu_list = fits.open(file_path, mode='readonly')
        hdu = hdu_list[0]
        return hdu.data, hdu.header
