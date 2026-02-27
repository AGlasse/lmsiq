import pickle

from astropy.io import fits
from astropy.io.fits import Card
import numpy as np
from lms_globals import Globals


class Transform:

    kw_comments = {'opticon': 'LMS wavelength coverage (nominal or extended)',
                   'slice_no': 'LMS spatial IFU slice number',
                   'spifu_no': 'LMS spectral IFU slice number',
                   'pri_ang': 'Prism rotation angle / deg.',
                   'ech_ang': 'Echelle rotation angle / deg.',
                   'ech_ord': 'Echelle diffraction order',
                   'w_min': 'Minimum slice wavelength (micron)',
                   'w_max': 'Maximum slice wavelength (micron)',
                   'n_mats': 'No of matrices (A, B, AI, BI)',
                   'mat_ord': 'Order of transform (4)'
                   }

    def __init__(self, **kwargs):
        self.matrices = {'a': None, 'b': None, 'ai': None, 'bi': None}
        self.lms_configuration = kwargs.get('lms_config', Globals.lms_config_template.copy())
        self.slice_configuration = kwargs.get('slice_config', Globals.slice_config_template.copy())
        if 'hdu_list' in kwargs:
            hdu_list = kwargs.get('hdu_list')
            ext_no = kwargs.get('ext_no')
            self.ingest_from_hdu_list(hdu_list, ext_no)
        if 'matrices' in kwargs:
            self.matrices = kwargs.get('matrices')
        return

    def ingest_from_hdu_list(self, hdu_list, ext_no):
        primary_hdr = hdu_list[0].header
        self.lms_configuration = Globals.lms_config_template.copy()
        for key in self.lms_configuration:          # Get slice 'common' parameters from primary header.
            self.lms_configuration[key] = primary_hdr[key.upper()]
        hdu = hdu_list[ext_no]
        for key in self.slice_configuration:
            self.slice_configuration[key] = hdu.header[key.upper()]
        self.read_matrices(hdu.data)
        return

    def is_match(self, slice_filter):
        slice_no_match = self.slice_configuration['slice_no'] == slice_filter['slice_no']
        spifu_no_match = self.slice_configuration['spifu_no'] == slice_filter['spifu_no']
        ech_ord_match = self.slice_configuration['ech_ord'] == slice_filter['ech_ord']
        is_match = slice_no_match and spifu_no_match and ech_ord_match
        return is_match

    def make_hdu_primary(self):
        lms_config = self.lms_configuration            # Get basic configuration parameters
        lms_config['n_mats'] = 4                       # and add useful header information.
        lms_config['mat_ord'] = 4
        cards = []
        for key in lms_config:
            if key in ['slice_no', 'spifu_no', 'ech_ord']:          # Exclude slice numbers from SVD fits primary
                continue
            card = Card(key.upper(), lms_config[key], Transform.kw_comments[key])
            cards.append(card)
        trace_hdr = fits.Header(cards)
        primary_hdu = fits.PrimaryHDU(header=trace_hdr)
        return primary_hdu

    def make_hdu_ext(self):
        mats = self.matrices
        a, b, ai, bi = mats['a'], mats['b'], mats['ai'], mats['bi']
        col_a = fits.Column(name='A', array=a.flatten(), format='E')
        col_b = fits.Column(name='B', array=b.flatten(), format='E')
        col_ai = fits.Column(name='AI', array=ai.flatten(), format='E')
        col_bi = fits.Column(name='BI', array=bi.flatten(), format='E')

        keys = Globals.slice_config_template.keys()
        sli_config = self.slice_configuration
        cards = []
        for key in keys:
            val = sli_config[key]
            card = Card(key.upper(), val, self.kw_comments[key])
            cards.append(card)
        hdr = fits.Header(cards)
        hdu_name = "SLICE_{:d}_{:d}".format(sli_config['slice_no'], sli_config['spifu_no'])
        bintable_hdu = fits.BinTableHDU.from_columns([col_a, col_b, col_ai, col_bi],
                                                     header=hdr, name=hdu_name)
        return bintable_hdu

    def read_matrices(self, table):
        """ Read the transform matrices from a fits hdu table object.
        """
        mat_order = Globals.svd_order
        mat_shape = mat_order, mat_order
        for mat_name in self.matrices:
            col_data = table[mat_name.upper()]
            matrix = np.reshape(col_data, mat_shape)
            self.matrices[mat_name] = matrix
        return

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
