import pickle
from astropy.io import fits
from astropy.io.fits import Card
import numpy as np


class Transform:

    config_common_kws = ['opticon', 'cfg_id', 'pri_ang', 'ech_ang',
                         'n_mats', 'mat_order', 'w_min', 'w_max']
    slice_specific_kws = ['slice_no', 'spifu_no', 'ech_order']
    kw_comments = {'opticon': 'LMS wavelength coverage mode',
                   'cfg_id': 'LMS mechanism configuration identifier',
                   'pri_ang': 'Prism rotation angle / deg.',
                   'ech_ang': 'Echelle rotation angle / deg.',
                   'ech_order': 'Echelle diffraction order number',
                   'n_mats': 'No. of distortion matrices (4)',
                   'mat_order': 'Matrix order (4x4)',
                   'slice_no': 'Spatial slice number',
                   'spifu_no': 'Spectral IFU slice no. =0 if not selected',
                   'w_min': 'Minimum slice wavelength (micron)',
                   'w_max': 'Maximum slice wavelength (micron)'
                   }

    def __init__(self, **kwargs):
        self.configuration = {}
        for kw in Transform.config_common_kws:
            self.configuration[kw] = None
        for kw in Transform.slice_specific_kws:
            self.configuration[kw] = None
        self.matrices = {'a': None, 'b': None, 'ai': None, 'bi': None}
        if 'cfg' in kwargs:         # Set any passed parameters.
            init_cfg = kwargs.get('cfg', {})
            for key in init_cfg:
                self.configuration[key] = init_cfg[key]
        if 'matrices' in kwargs:
            self.matrices = kwargs.get('matrices')
        return

    def make_hdu_primary(self):
        cfg = self.configuration
        cards = []
        for key in cfg:
            if key in Transform.slice_specific_kws:  # Skip slice specific key words
                continue
            card = Card(key.upper(), cfg[key], Transform.kw_comments[key])
            cards.append(card)
        trace_hdr = fits.Header(cards)
        primary_hdu = fits.PrimaryHDU(header=trace_hdr)
        return primary_hdu

    def make_hdu_ext(self):
        mats = self.matrices
        cfg = self.configuration
        a, b, ai, bi = mats['a'], mats['b'], mats['ai'], mats['bi']
        col_a = fits.Column(name='A', array=a.flatten(), format='E')
        col_b = fits.Column(name='B', array=b.flatten(), format='E')
        col_ai = fits.Column(name='AI', array=ai.flatten(), format='E')
        col_bi = fits.Column(name='BI', array=bi.flatten(), format='E')

        cards = []
        for key in Transform.slice_specific_kws:
            card = Card(key.upper(), cfg[key], self.kw_comments[key])
            cards.append(card)
        hdr = fits.Header(cards)
        hdu_name = "SLICE_{:d}_{:d}".format(cfg['slice_no'], cfg['spifu_no'])
        bintable_hdu = fits.BinTableHDU.from_columns([col_a, col_b, col_ai, col_bi],
                                                     header=hdr, name=hdu_name)
        return bintable_hdu

    @staticmethod
    def decode_primary_hdr(primary_hdr):
        pri_cfg = {}
        for key in Transform.config_common_kws:
            pri_cfg[key] = primary_hdr[key]
        return pri_cfg

    def ingest_extension_hdr(self, hdr):
        for key in Transform.slice_specific_kws:
            self.configuration[key] = hdr[key]
        return

    def read_matrices(self, table):
        """ Read the transform matrices from a fits hdu table object.
        """
        mat_order = self.configuration['mat_order']
        mat_shape = mat_order, mat_order
        for mat_name in self.matrices:
            col_data = table[mat_name.upper()]
            matrix = np.reshape(col_data, mat_shape)
            self.matrices[mat_name] = matrix
        return

        #
        # a_col, b_col = table.field('a'), table.field('b')
        # ai_col, bi_col = table.field('ai'), table.field('bi')
        # a = np.reshape(a_col, mat_shape)
        # b = np.reshape(b_col, mat_shape)
        # ai = np.reshape(ai_col, mat_shape)
        # bi = np.reshape(bi_col, mat_shape)
        # matrices = {'a': a, 'b': b, 'ai': ai, 'bi': bi}
        # transform = {'configuration': config, 'matrices': matrices}

        # transform_list.append(transform)
        # hdu_list.close()
        # return transform_list

    # def read_fits(self, fits_name):
    #     """ Read fits file into a list of 'transform' objects, one per spatial/spectral slice combination.
    #     """
    #     fits_path = self.tf_dir + fits_name
    #     hdu_list = fits.open(fits_path, mode='readonly')
    #     primary_hdr = hdu_list[0].header
    #     opticon = primary_hdr['OPTICON']
    #     ea = primary_hdr['ECH_ANG']
    #     pa = primary_hdr['PRI_ANG']
    #     n_mats = primary_hdr['N_MATS']
    #     n_mat_terms = primary_hdr['N_TERMS']
    #     w_min = primary_hdr['W_MIN']
    #     w_max = primary_hdr['W_MAX']
    #
    #     base_config = {'opticon': opticon, 'ech_ang': ea, 'pri_ang': pa,
    #                    'n_mats': n_mats, 'n_mat_terms': n_mat_terms,
    #                    'w_min': w_min, 'w_max': w_max}
    #     mat_shape = n_mat_terms, n_mat_terms
    #
    #     transform_list = []
    #     tr_hdr_keys = ['ECH_ORD', 'SLICE', 'SPIFU', 'XP1', 'XP3', 'XP5', 'YP1', 'YP3', 'YP5']
    #     for hdu in hdu_list[1:]:
    #         table, hdr = hdu.data, hdu.header
    #         config = base_config.copy()
    #         for key in tr_hdr_keys:
    #             config[key.lower()] = hdr[key]
    #
    #         a_col, b_col = table.field('a'), table.field('b')
    #         ai_col, bi_col = table.field('ai'), table.field('bi')
    #         a = np.reshape(a_col, mat_shape)
    #         b = np.reshape(b_col, mat_shape)
    #         ai = np.reshape(ai_col, mat_shape)
    #         bi = np.reshape(bi_col, mat_shape)
    #         matrices = {'a': a, 'b': b, 'ai': ai, 'bi': bi}
    #         transform = {'configuration': config, 'matrices': matrices}
    #         transform_list.append(transform)
    #     hdu_list.close()
    #     return transform_list

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
