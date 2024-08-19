import os
from os import listdir
import shutil
import pickle
from astropy.io import fits
from astropy.io.fits import Card, HDUList
from lms_globals import Globals
import numpy as np
import lmsdist_trace


class Transform:

    def __init__(self):
        self.configuration = {}
        self.matrices = {}
        return

    def write_fits(self, trace):
        # Create data tables holding transforms for all slices
        par = trace.parameter
        ea = par['Echelle angle']
        pa = par['Prism angle']
        opticon = 'Extended' if trace.is_spifu else 'Nominal'
        n_mats = Globals.n_mats_transform
        n_mat_terms = trace.n_mat_terms

        primary_cards = [Card('OPTICON', opticon, 'Optical configuration'),
                         Card('ECH_ANG', ea, 'Echelle angle / deg'),
                         Card('PRI_ANG', pa, 'Prism angle / deg'),
                         Card('N_MATS', n_mats, 'A, B, AI, BI transform matrices'),
                         Card('N_TERMS', n_mat_terms, 'Transform matrix dimensions'),
                         Card('W_MIN', trace.wmin, 'Short wavelength limit / micron'),
                         Card('W_MAX', trace.wmax, 'Long wavelength limit / micron')
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
            ech_order, slice_no, spifu_no = slice_object[0]
            cards = [Card('ECH_ORD', ech_order, 'Echelle diffraction order'),
                     Card('SLICE', slice_no, 'Spatial slice number (1 <= slice_no <= 28)'),
                     Card('SPIFU', spifu_no, 'Spectral IFU slice no.')]

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

    def read_fits(self, fits_name):
        """ Read fits file into a 'trace' object, which
        """
        fits_path = self.tf_dir + fits_name
        hdu_list = fits.open(fits_path, mode='readonly')
        primary_hdr = hdu_list[0].header
        opticon = primary_hdr['OPTICON']
        ea = primary_hdr['ECH_ANG']
        pa = primary_hdr['PRI_ANG']
        n_mats = primary_hdr['N_MATS']
        n_mat_terms = primary_hdr['N_TERMS']
        w_min = primary_hdr['W_MIN']
        w_max = primary_hdr['W_MAX']

        base_config = {'opticon': opticon, 'ech_ang': ea, 'pri_ang': pa,
                       'n_mats': n_mats, 'n_mat_terms': n_mat_terms,
                       'w_min': w_min, 'w_max': w_max}
        mat_shape = n_mat_terms, n_mat_terms

        transform_list = []
        tr_hdr_keys = ['ECH_ORD', 'SLICE', 'SPIFU', 'XP1', 'XP3', 'XP5', 'YP1', 'YP3', 'YP5']
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
