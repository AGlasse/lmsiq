import os
from os import listdir
import shutil
import pickle
from astropy.io import fits
from astropy.io.fits import Card, HDUList
from lms_globals import Globals
import numpy as np
import lmsdist_trace


class Filer:

    analysis_types = ['distortion', 'iq']
    trace_file, poly_file, wcal_file, stats_file, tf_fit_file = None, None, None, None, None
    base_results_path = None
    cube_folder, iq_png_folder = None, None
    slice_results_path, dataset_results_path = None, None
    pdp_path, profiles_path, centroids_path = None, None, None

    def __init__(self, model_configuration):
        analysis_type, optical_path, date_stamp, _, _, _ = model_configuration
        self.model_configuration = model_configuration
        sub_folder = "{:s}/{:s}/{:s}".format(analysis_type, optical_path, date_stamp)
        self.data_folder = self.get_folder('../data/' + sub_folder)
        self.output_folder = self.get_folder('../output/' + sub_folder)
        file_leader = self.output_folder + sub_folder.replace('/', '_')
        if analysis_type == 'distortion':
            self.tf_dir = self.get_folder(self.output_folder + 'fits')
            self.trace_file = file_leader + '_trace'  # All ray coordinates
            self.poly_file = file_leader + '_dist_poly.txt'
            self.wcal_file = file_leader + '_dist_wcal.txt'  # Echelle angle as function of wavelength
            self.stats_file = file_leader + '_dist_stats.txt'
            self.tf_fit_file = file_leader + '_dist_tf_fit'  # Use pkl files to write objects directly
        if analysis_type == 'iq':
            self.cube_folder = self.get_folder(self.output_folder + '/cube')
        return

    @staticmethod
    def get_file_list(folder, inc_tags=[], exc_tags=[]):
        file_list = listdir(folder)
        for tag in inc_tags:
            file_list = [f for f in file_list if tag in f]
        for tag in exc_tags:
            file_list = [f for f in file_list if tag not in f]
        return file_list

    @staticmethod
    def get_folder(in_path):
        tokens = in_path.split('/')
        out_path = ''
        for token in tokens:
            out_path = out_path + token + '/'
            if not os.path.exists(out_path):
                os.mkdir(out_path)
        return out_path

    def write_fits_transform(self, trace):
        # Create data tables holding transforms for all slices
        par = trace.parameter
        ea = par['Echelle angle']
        pa = par['Prism angle']
        opticon = trace.opticon
        trc = Globals.transform_config
        n_mats = trc['n_mats']
        mat_order = trc['mat_order']

        primary_cards = [Card('OPTICON', opticon, 'Optical configuration'),
                         Card('ECH_ANG', ea, 'Echelle angle / deg'),
                         Card('PRI_ANG', pa, 'Prism angle / deg'),
                         Card('N_MATS', n_mats, 'A, B, AI, BI transform matrices'),
                         Card('MAT_ORD', mat_order, 'Transform matrix dimensions')
                         ]
        trace_hdr = fits.Header(primary_cards)        # {'ECH_ANG': ea}
        primary_hdu = fits.PrimaryHDU(header=trace_hdr)
        hdu_list = HDUList([primary_hdu])
        # Create fits file with primaryHDU only
        ea_tag = self._make_fits_tag(ea)
        pa_tag = self._make_fits_tag(pa)
        fmt = "lms_dist_ea_{:s}_pa_{:s}"
        fits_name = fmt.format(ea_tag, pa_tag)
        fits_path = self.tf_dir + fits_name + '.fits'
        for slice_object in trace.slice_objects:
            ech_order, slice_no, spifu_no, w_min, w_max = slice_object[0]
            cards = [Card('ECH_ORD', ech_order, 'Echelle diffraction order'),
                     Card('SLICE', slice_no, 'Spatial slice number (1 <= slice_no <= 28)'),
                     Card('SPIFU', spifu_no, 'Spectral IFU slice no.'),
                     Card('W_MIN', w_min, 'Minimum slice wavelength (micron)'),
                     Card('W_MAX', w_max, 'Maximum slice wavelength (micron)')]

            a, b, ai, bi = slice_object[1]
            col_a = fits.Column(name='a', array=a.flatten(), format='E')
            col_b = fits.Column(name='b', array=b.flatten(), format='E')
            col_ai = fits.Column(name='ai', array=ai.flatten(), format='E')
            col_bi = fits.Column(name='bi', array=bi.flatten(), format='E')

            xpoly_corr, ypoly_corr = slice_object[2]            # x and y polynomial correction factors
            cards.append(Card('XP1', xpoly_corr[0], 'dx = x^1 XP1 + x^3 XP3 + x^5 XP5'))
            cards.append(Card('XP3', xpoly_corr[1], 'dx = x^1 XP1 + x^3 XP3 + x^5 XP5'))
            cards.append(Card('XP5', xpoly_corr[2], 'dx = x^1 XP1 + x^3 XP3 + x^5 XP5'))
            cards.append(Card('YP1', ypoly_corr[0], 'dy = x^1 YP1 + x^3 YP3 + x^5 YP5'))
            cards.append(Card('YP3', ypoly_corr[1], 'dy = x^1 YP1 + x^3 YP3 + x^5 YP5'))
            cards.append(Card('YP5', ypoly_corr[2], 'dy = x^1 YP1 + x^3 YP3 + x^5 YP5'))

            hdr = fits.Header(cards)
            bintable_hdu = fits.BinTableHDU.from_columns([col_a, col_b, col_ai, col_bi], header=hdr)
            hdu_list.append(bintable_hdu)

        hdu_list.writeto(fits_path, overwrite=True)
        return fits_name

    def read_fits_transforms(self):
        fits_file_list = Filer.get_file_list(self.tf_dir, inc_tags=[], exc_tags=[])
        transform_list = []
        for fits_name in fits_file_list:
            transforms = self.read_fits_transform(fits_name)
            for transform in transforms:
                transform_list.append(transform)
        return transform_list

    def read_fits_transform(self, fits_name):
        """ Read fits file into a 'trace' object, which
        """
        fits_path = self.tf_dir + fits_name
        hdu_list = fits.open(fits_path, mode='readonly')
        primary_hdr = hdu_list[0].header
        opticon = primary_hdr['OPTICON']
        ea = primary_hdr['ECH_ANG']
        pa = primary_hdr['PRI_ANG']
        n_mats = primary_hdr['N_MATS']
        mat_order = primary_hdr['MAT_ORD']

        base_config = {'opticon': opticon, 'ech_ang': ea, 'pri_ang': pa,
                       'n_mats': n_mats, 'mat_order': mat_order}
        mat_shape = mat_order, mat_order

        transform_list = []
        tr_hdr_keys = ['ECH_ORD', 'SLICE', 'SPIFU', 'W_MIN', 'W_MAX',
                       'XP1', 'XP3', 'XP5', 'YP1', 'YP3', 'YP5']
        for hdu in hdu_list[1:]:
            table, hdr = hdu.data, hdu.header
            config = base_config.copy()
            for key in tr_hdr_keys:
                config[key.lower()] = hdr[key]

            a_col, b_col = table.field('a'), table.field('b')
            ai_col, bi_col = table.field('ai'), table.field('bi')
            a = np.reshape(a_col, mat_shape)
            b = np.reshape(b_col, mat_shape)
            ai = np.reshape(ai_col, mat_shape)
            bi = np.reshape(bi_col, mat_shape)
            matrices = {'a': a, 'b': b, 'ai': ai, 'bi': bi}
            transform = {'configuration': config, 'matrices': matrices}

            transform_list.append(transform)
        hdu_list.close()
        return transform_list

    def _make_fits_tag(self, val):
        val_str = "{:5.3f}".format(val)
        tag = val_str.replace('-', 'm')
        tag = tag.replace('.','p')
        return tag

    @staticmethod
    def read_pickle(pickle_path):
        if pickle_path[-4:] != '.pkl':
            pickle_path += '.pkl'
        file = open(pickle_path, 'rb')
        python_object = pickle.load(file)
        file.close()
        return python_object

    @staticmethod
    def write_pickle(pickle_path, python_object):
        file = open(pickle_path + '.pkl', 'wb')
        pickle.dump(python_object, file)
        file.close()
        return

    def _get_results_path(self, data_id, data_type):
        dataset, slice_subfolder, ipc_tag, process_level, config_folder, mcrun_tag, axis = data_id
        folder = self.output_folder + slice_subfolder
        folder += ipc_tag + '/' + process_level + '/' + data_type
        folder = self.get_folder(folder)

        type_tags = {'xcentroids': '_xcen', 'ycentroids': '_ycen',
                     'xfwhm_gau': '_xfwhm', 'photometry': '_phot',
                     'ee_spectral': '_eex', 'ee_spatial': '_eey',
                     'lsf_spectral': '_lsx', 'lsf_spatial': '_lsy',
                     'ee_dfp_spectral': '_eex_dfp', 'ee_dfp_spatial': '_eey_dfp',
                     'lsf_dfp_spectral': '_lsx_dfp', 'lsf_dfp_spatial': '_lsy_dfp',
                     }
        type_tag = type_tags[data_type]
        config_tag = config_folder[0:-1]
        slice_tag = slice_subfolder[:-1] + '_'
        file_name = slice_tag + ipc_tag + '_' + process_level + type_tag + '_wav_' + config_tag + '.csv'
        path = folder + file_name
        return path
