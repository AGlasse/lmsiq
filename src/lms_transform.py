import pickle
from astropy.io import fits
from astropy.io.fits import Card
import numpy as np
from lms_globals import Globals


class Transform:

    kw_comments = {'opticon': 'LMS wavelength coverage (nominal or extended)',
                   'pri_ang': 'Prism rotation angle / deg.',
                   'ech_ang': 'Echelle rotation angle / deg.',
                   'n_mats': 'No. of distortion matrices (4)',
                   'mat_order': 'Matrix order (4x4)',
                   'w_min': 'Minimum slice wavelength (micron)',
                   'w_max': 'Maximum slice wavelength (micron)'
                   }

    def __init__(self, **kwargs):
        self.matrices = {'a': None, 'b': None, 'ai': None, 'bi': None}
        self.configuration = {}
        for key in self.kw_comments:
            self.configuration[key] = None
        self.slice_no, self.spifu_no, self.ech_order = None, None, None
        if 'trace' in kwargs:               # Get transform from Trace object
            self.ingest_from_trace(**kwargs)
        if 'hdu_list' in kwargs:
            hdu_list = kwargs.get('hdu_list')
            ext_no = kwargs.get('ext_no')
            self.ingest_from_hdu_list(hdu_list, ext_no)
        if 'matrices' in kwargs:
            self.matrices = kwargs.get('matrices')
        self.slice_configuration = {'slice_no': self.slice_no, 'spifu_no': self.spifu_no, 'ech_order': self.ech_order}
        return

    def ingest_from_trace(self, **kwargs):
        trace = kwargs.get('trace')
        self.slice_no = kwargs.get('slice_no')
        self.spifu_no = kwargs.get('spifu_no')
        self.ech_order = kwargs.get('ech_order')

        for key in trace.lms_config:
            self.configuration[key] = trace.lms_config[key]
        return

    def make_hdu_primary(self, w_min, w_max):
        cfg = self.configuration
        cfg['n_mats'] = 4
        cfg['mat_order'] = 4
        cfg['w_min'] = w_min
        cfg['w_max'] = w_max
        cards = []
        for key in cfg:
            if key in ['slice_no', 'spifu_no', 'ech_order']:          # Exclude slice numbers from SVD fits primary
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


        keys = ['slice_no', 'spifu_no', 'ech_order']
        vals = [self.slice_no, self.spifu_no, self.ech_order]
        comments = ['Nominal config. slice number', 'Extended config. spectral slice number',
                    'Echelle diffraction order']
        cards = []
        for i in range(0, len(keys)):
            card = Card(keys[i].upper(), vals[i], comments[i])
            cards.append(card)
        hdr = fits.Header(cards)
        hdu_name = "SLICE_{:d}_{:d}".format(self.slice_no, self.spifu_no)
        bintable_hdu = fits.BinTableHDU.from_columns([col_a, col_b, col_ai, col_bi],
                                                     header=hdr, name=hdu_name)
        return bintable_hdu

    def ingest_from_hdu_list(self, hdu_list, ext_no):
        primary_hdr = hdu_list[0].header
        for key in self.configuration:          # Get slice 'common' parameters from primary header.
            self.configuration[key] = primary_hdr[key.upper()]
        hdu = hdu_list[ext_no]
        self.slice_no = hdu.header['slice_no'.upper()]
        self.spifu_no = hdu.header['spifu_no'.upper()]
        self.ech_order = hdu.header['ech_order'.upper()]
        self.read_matrices(hdu.data)
        return

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
