import os
from os import listdir
import pickle
from astropy.io import fits
from astropy.io.fits import Card, HDUList
from lms_globals import Globals
import numpy as np


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

    def write_fits_affine_transform(self, trace):
        _, _, date_stamp, _, _, _ = trace.model_configuration
        affines = trace.affines
        n_fwds, mat_order, _ = affines.shape
        affine_inverses = np.linalg.inv(affines)

        primary_cards = [Card('N_MATS', 2*n_fwds, 'MFP <-> DFP transform matrices'),
                         Card('MAT_ORD', mat_order, 'Transform matrix dimensions')
                         ]
        trace_hdr = fits.Header(primary_cards)
        primary_hdu = fits.PrimaryHDU(header=trace_hdr)
        hdu_list = HDUList([primary_hdu])

        fmt = "lms_dist_mfp_dfp_v{:s}"
        fits_name = fmt.format(date_stamp)
        fits_path = self.tf_dir + fits_name + '.fits'
        cards = []

        col_list = []
        for m in range(0, n_fwds):
            col_name = "MFP>D{:d}".format(m)
            col = fits.Column(name=col_name, array=affines[m].flatten(), format='E')
            col_list.append(col)
        for m in range(0, n_fwds):
            col_name = "D{:d}>MFP".format(m)
            col = fits.Column(name=col_name, array=affine_inverses[m].flatten(), format='E')
            col_list.append(col)

        hdr = fits.Header(cards)
        bintable_hdu = fits.BinTableHDU.from_columns(col_list, header=hdr)
        hdu_list.append(bintable_hdu)
        hdu_list.writeto(fits_path, overwrite=True)
        return

    def read_fits_affine_transform(self, date_stamp):
        """ Read fits file into a 'trace' object, which
        """
        fmt = "lms_dist_mfp_dfp_v{:s}"

        fits_name = fmt.format(date_stamp)
        fits_path = self.tf_dir + fits_name + '.fits'
        hdu_list = fits.open(fits_path, mode='readonly')
        primary_hdr = hdu_list[0].header
        n_mats = primary_hdr['N_MATS']
        mat_order = primary_hdr['MAT_ORD']
        aff_shape = n_mats, mat_order, mat_order
        affines = np.zeros(aff_shape)
        hdu = hdu_list[1]

        table, hdr = hdu.data, hdu.header
        n_cols = len(hdu.columns)
        for i in range(0, n_cols):
            col_vals = table.field(i)
            matrix = np.reshape(col_vals, (mat_order, mat_order))
            affines[i] = matrix
        return affines

    def write_fits_svd_transform(self, trace):
        # Create data tables holding transforms for all slices
        pa = trace.parameter['Prism angle']
        ea = trace.parameter['Echelle angle']
        _, opticon, date_stamp, _, _, _ = trace.model_config
        trc = Globals.transform_config
        n_mats = trc['n_mats']
        mat_order = trc['mat_order']

        primary_cards = [Card('OPTICON', opticon, 'Optical configuration'),
                         Card('PRI_ANG', pa, 'Prism angle / deg'),
                         Card('ECH_ANG', ea, 'Echelle angle / deg'),
                         Card('N_MATS', n_mats, 'A, B, AI, BI transform matrices'),
                         Card('MAT_ORD', mat_order, 'Transform matrix dimensions')
                         ]
        trace_hdr = fits.Header(primary_cards)        # {'ECH_ANG': ea}
        primary_hdu = fits.PrimaryHDU(header=trace_hdr)
        hdu_list = HDUList([primary_hdu])
        # Create fits file with primaryHDU only
        otag = '_nom' if opticon == 'nominal' else '_ext'
        ptag = "_pa{:05d}".format(abs(int(10000. * pa)))
        esign = 'p' if ea > 0. else 'n'
        etag = "_ea{:s}{:05d}".format(esign, abs(int(10000. * ea)))
        vtag = "_v{:s}".format(date_stamp)
        fmt = "lms_efp_mfp{:s}{:s}{:s}{:s}"
        fits_name = fmt.format(otag, ptag, etag, vtag)
        fits_path = self.tf_dir + fits_name + '.fits'
        for ifu_slice in trace.slices:
            ech_order, slice_no, spifu_no, w_min, w_max = ifu_slice[0]
            cards = [Card('ECH_ORD', ech_order, 'Echelle diffraction order'),
                     Card('SLICE', slice_no, 'Spatial slice number (1 <= slice_no <= 28)'),
                     Card('SPIFU', spifu_no, 'Spectral IFU slice no.'),
                     Card('W_MIN', w_min, 'Minimum slice wavelength (micron)'),
                     Card('W_MAX', w_max, 'Maximum slice wavelength (micron)')]

            a, b, ai, bi = ifu_slice[1]
            col_a = fits.Column(name='a', array=a.flatten(), format='E')
            col_b = fits.Column(name='b', array=b.flatten(), format='E')
            col_ai = fits.Column(name='ai', array=ai.flatten(), format='E')
            col_bi = fits.Column(name='bi', array=bi.flatten(), format='E')

            hdr = fits.Header(cards)
            bintable_hdu = fits.BinTableHDU.from_columns([col_a, col_b, col_ai, col_bi], header=hdr)
            hdu_list.append(bintable_hdu)

        hdu_list.writeto(fits_path, overwrite=True)
        return fits_name

    def read_fits_svd_transform(self, fits_name):
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
        tr_hdr_keys = (['ECH_ORD', 'SLICE', 'SPIFU', 'W_MIN', 'W_MAX'])
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

    def read_fits_svd_transforms(self):
        inc_tags = ['efp_mfp']
        fits_file_list = Filer.get_file_list(self.tf_dir, inc_tags=inc_tags, exc_tags=[])
        transform_list = []
        for fits_name in fits_file_list:
            transforms = self.read_fits_svd_transform(fits_name)
            for transform in transforms:
                transform_list.append(transform)
        return transform_list

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
