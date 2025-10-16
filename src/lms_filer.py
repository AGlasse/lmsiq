import os
from os import listdir
import pickle
from astropy.table import Table
from astropy.io import fits
from astropy.io.fits import Card, HDUList, ImageHDU, PrimaryHDU
from lms_globals import Globals
from lms_transform import Transform
import numpy as np


class Filer:

    analysis_types = ['distortion', 'iq']
    trace_file, poly_file, wcal_file, stats_file, tf_fit_file = None, None, None, None, None
    base_results_path = None
    cube_folder, iq_png_folder = None, None
    slice_results_path, dataset_results_path = None, None
    pdp_path, profiles_path, centroids_path = None, None, None
    test_data_folder = None

    def __init__(self, model_configuration):
        analysis_type, opticon, date_stamp, _, _, _ = model_configuration
        # Filer.wpa_fit_order = Globals.wpa_fit_order_dict[opticon]
        self.model_configuration = model_configuration
        sub_folder = "{:s}/{:s}/{:s}".format(analysis_type, opticon, date_stamp)
        self.data_folder = self.get_folder('../data/model/' + sub_folder)
        self.sim_folder = self.get_folder('../data/sim/' + sub_folder)
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
    def read_zemax_fits(path, pri_hdr_ext=0, data_exts=[0]):
        """ Read in a Zemax model (PSF) image.
        """
        hdu_list = fits.open(path, mode='readonly')
        pri_hdr = hdu_list[pri_hdr_ext].header
        image_hdus = [hdu_list[i] for i in data_exts]
        if len(data_exts) == 1:
            return pri_hdr, image_hdus[0]
        return pri_hdr, image_hdus

    @staticmethod
    def write_zemax_fits(path, header, data):
        """ Write a Zemax image.  Header in hdu[0].  If data is a (3D) image stack, they are written to hdu[1] onwards,
        while a single image is written to hdu[0]
        """
        n_frames = len(data)
        primary_hdu = PrimaryHDU(header=header)
        hdu_list = HDUList([primary_hdu])
        for i in range(0, n_frames):
            if n_frames == 1:
                hdu_list[0].data = data
                break
            hdu = ImageHDU(data[i])
            hdu.name = "Detector {:d}".format(i+1)
            hdu_list.append(hdu)
        hdu_list.writeto(path, overwrite=True, checksum=True)
        return

    @staticmethod
    def set_test_data_folder(simulator_name):
        Filer.test_data_folder = "../data/test_{:s}".format(simulator_name)
        return

    @staticmethod
    def read_mosaics(inc_tags=[], exc_tags=[]):
        mosaics = []
        data_ext_nos = [2, 1, 3, 4]
        folder = Filer.test_data_folder
        file_list = Filer.get_file_list(folder, inc_tags=inc_tags, exc_tags=exc_tags)
        if len(file_list) == 0:
            text = "Files in {:s} including tags ".format(folder)
            for tag in inc_tags:
                text += "{:s}, ".format(tag)
            text += 'not found'
            return mosaics

        for file in file_list:
            path = folder + '/' + file
            hdr, data = Filer.read_zemax_fits(path, data_exts=data_ext_nos)
            mosaic = file, hdr, data
            mosaics.append(mosaic)
        return mosaics

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

    def write_fit_parameters(self, wpa_fit, wxo_fit, wxo_hdr, term_fits):

        _, opticon, date_stamp, _, _, _ = self.model_configuration
        fmt = "lms_dist_efp_mfp_{:s}_fit_parameters_v{:s}"
        fits_name = fmt.format(opticon[0:3], date_stamp)
        fits_path = self.tf_dir + fits_name + '.fits'

        n_slices = len(term_fits)
        primary_cards = [Card('OPTICON', opticon, 'Optical configuration (nominal/extended)')]
        primary_header = fits.Header(primary_cards)
        primary_hdu = fits.PrimaryHDU(header=primary_header)
        hdu_list = HDUList([primary_hdu])

        wpa_col_names = []
        for i in range(0, wpa_fit['n_coefficients']):
            wpa_col_names.append("WP{:d}".format(i))
        wpa_data = wpa_fit['wpa_opt']     # All slices have the same wxo fit parameters...
        wpa_table = Table(data=np.array(wpa_data), names=wpa_col_names)
        n_coeffs = wpa_fit['n_coefficients']
        wpa_cards = [Card('DESCR', 'Fit parms to map wavelength to prism angle', ''),
                     Card('N_COEFFS', n_coeffs, 'No. of polynomial fit coefficients')
                     ]
        wpa_hdr = fits.Header(wpa_cards)

        wpa_hdu = fits.BinTableHDU(data=wpa_table, header=wpa_hdr)
        hdu_list.append(wpa_hdu)

        wxo_column_names = ['SLICE_NO', 'SPIFU_NO'] + wxo_hdr
        n_columns = len(wxo_column_names)
        wxo_data = np.zeros((n_columns))
        # Write wavelength x echelle order fit parameters to second HDU
        wxo_data[0] = wxo_fit['slice_no']
        wxo_data[1] = wxo_fit['spifu_no']
        wxo_data[2:n_columns] = wxo_fit['wxo_opt']     # All slices have the same wxo fit parameters...
        order = wxo_fit['order']
        n_coeffs = wxo_fit['n_coefficients']
        wxo_cards = [Card('DESCR', 'Fit parms to map prism and echelle angle to slice 13 wavelength', ''),
                     Card('ORDER', order, 'Surface fit matrix order'),
                     Card('N_COEFFS', n_coeffs, 'No. of coefficients in surface fit matrix')
                     ]
        wxo_hdr = fits.Header(wxo_cards)
        wxo_table = Table(data=wxo_data, names=wxo_column_names)
        wxo_hdu = fits.BinTableHDU(data=wxo_table, header=wxo_hdr)
        hdu_list.append(wxo_hdu)

        term_cards = [Card('DESCR', 'Fit parms to map prism and echelle angle to transform terms.', ''),
                      Card('ORDER', order, 'Surface fit matrix order'),
                      Card('N_COEFFS', n_coeffs, 'No. of coefficients in surface fit matrix')
                      ]
        term_hdr = fits.Header(term_cards)
        mat_tags_uc = ['A', 'B', 'AI', 'BI']
        term_names = ['SLICE_NO', 'SPIFU_NO', 'ROW', 'COL']
        for mat_tag in mat_tags_uc:
            for tag in wxo_column_names[2:]:
                term_names.append(mat_tag + '_' + tag)
        n_columns = len(term_names)
        n_records_slice = Globals.svd_order * Globals.svd_order
        term_data = np.zeros((n_slices * n_records_slice, n_columns))
        data_row = 0                           # Row counter in term_data array
        for i, term_row in enumerate(term_fits):
            matrices = term_row[2]
            nr, nc, nvals = matrices['a'].shape
            nrc = nr * nc
            term_data[data_row:data_row + nrc, 0:2] = term_row[0:2]     # slice_no, spifu_no
            rows, cols = [], []
            for rc in range(0, nr*nc):
                rows.append(int(rc / nr))
                cols.append(int(rc % nc))
            term_data[data_row:data_row + nrc, 2] = rows  # matrix row and column nos.
            term_data[data_row:data_row + nrc, 3] = cols
            data_col = 4
            for key in matrices:
                data_row = i * nrc
                matrix = matrices[key]
                for row in range(0, nr):
                    data_block = matrix[row, :]
                    term_data[data_row:data_row+4, data_col:data_col+nvals] = data_block
                    data_row += nc
                data_col += nvals

        term_table = Table(data=term_data, names=term_names)
        term_hdu = fits.BinTableHDU(data=term_table, header=term_hdr)
        hdu_list.append(term_hdu)
        hdu_list.writeto(fits_path, overwrite=True)
        return

    def read_fit_parameters(self, opticon):
        """ Method to read the surface fit parameters from the fits file.
        """
        _, _, date_stamp, _, _, _ = self.model_configuration
        fmt = "lms_dist_efp_mfp_{:s}_fit_parameters_v{:s}"
        fits_name = fmt.format(opticon[0:3], date_stamp)
        fits_path = self.tf_dir + fits_name + '.fits'
        hdu_list = fits.open(fits_path, mode='readonly')


        wpa_data = hdu_list[1].data
        wxo_header = hdu_list[1].header
        n_fit_coeffs = wxo_header['N_COEFFS']
        wpa_fit = {'n_coeffs': n_fit_coeffs,  'opt': list(wpa_data[0][:])}

        wxo_data = hdu_list[2].data
        slice_no = int(wxo_data['SLICE_NO'][0])
        spifu_no = int(wxo_data['SPIFU_NO'][0])
        wxo_header = hdu_list[2].header
        fit_order, n_fit_coeffs = wxo_header['ORDER'], wxo_header['N_COEFFS']
        wxo_fit = {'order': fit_order, 'n_coeffs': n_fit_coeffs,
                   'slice_no': slice_no, 'spifu_no': spifu_no,
                   'opt': list(wxo_data[0][2:])}

        term_data = np.array(hdu_list[3].data)
        term_header = hdu_list[3].header
        svd_order = term_header['ORDER']
        n_fit_coeffs = term_header['N_COEFFS']
        mat_length = svd_order * svd_order
        mat_shape = svd_order, svd_order, n_fit_coeffs,
        term_fits = {}

        # Should have 18 elements for ext, 28 for nominal
        for data_rec in np.array(term_data):
            data_array = list(data_rec)
            # print(data_list[0:10])
            slice_no, spifu_no = int(data_array[0]), int(data_array[1])
            if slice_no not in term_fits.keys():
                term_fits[slice_no] = {}
            if spifu_no not in term_fits[slice_no].keys():
                term_fits[slice_no][spifu_no] = {}
                for mat_name in Globals.matrix_names:
                    term_fits[slice_no][spifu_no][mat_name] = np.zeros(mat_shape)
            mat_row, mat_col = int(data_array[2]), int(data_array[3])
            data_col = 4
            for mat_name in Globals.matrix_names:
                mat = term_fits[slice_no][spifu_no][mat_name]
                mat[mat_row, mat_col] = data_array[data_col: data_col + n_fit_coeffs]
                term_fits[slice_no][spifu_no][mat_name] = mat
                data_col += n_fit_coeffs
        return wpa_fit, wxo_fit, term_fits

    def write_affine_transform(self, trace):
        _, _, date_stamp, _, _, _ = trace.model_configuration
        affines = trace.affines
        n_mats, mat_order, _ = affines.shape

        primary_cards = [Card('N_MATS', n_mats, 'MFP <-> DFP transform matrices'),
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
        for m in range(0, n_mats):
            col_name = "MFP>D{:d}".format(m+1) if m < 4 else "D{:d}>MFP".format(m-3)
            col = fits.Column(name=col_name, array=affines[m].flatten(), format='E')
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

    def write_svd_transforms(self, trace):
        """ Create data tables holding transforms for all slices in a configuration and write them
        to a fits file.
        """
        hdu_list = None
        write_primary = True
        for transform in trace.transforms:
            cfg = transform.configuration
            if write_primary:
                primary_hdu = transform.make_hdu_primary()
                hdu_list = HDUList([primary_hdu])
                # hdu_list.append(primary_hdu)

                # Create fits file with primaryHDU only
                otag = '_nom' if cfg['opticon'] == 'nominal' else '_ext'
                ptag = "_pa{:05d}".format(abs(int(10000. * cfg['pri_ang'])))
                ea = cfg['ech_ang']
                esign = 'p' if ea > 0. else 'n'
                etag = "_ea{:s}{:05d}".format(esign, abs(int(10000. * ea)))
                _, _, date_stamp, _, _, _ = trace.model_config
                vtag = "_v{:s}".format(date_stamp)
                fmt = "lms_efp_mfp{:s}{:s}{:s}{:s}"
                fits_name = fmt.format(otag, ptag, etag, vtag)
                fits_path = self.tf_dir + fits_name + '.fits'
                write_primary = False
            bintable_hdu = transform.make_hdu_ext()
            hdu_list.append(bintable_hdu)
        hdu_list.writeto(fits_path, overwrite=True)
        return fits_name

    # def read_svd_transforms(self, fits_name):
    #     """ Read a list of transforms between the EFP and MFP from a fits file.
    #     """
    #     fits_path = self.tf_dir + fits_name
    #     hdu_list = fits.open(fits_path, mode='readonly')
    #     primary_hdr = hdu_list[0].header
    #     opticon = primary_hdr['OPTICON']
    #     cfg_id = primary_hdr['CFG_ID']
    #     ea = primary_hdr['ECH_ANG']
    #     pa = primary_hdr['PRI_ANG']
    #     n_mats = primary_hdr['N_MATS']
    #     w_min = primary_hdr['W_MIN']
    #     w_max = primary_hdr['W_MAX']
    #     mat_order = primary_hdr['MAT_ORDER']
    #
    #     base_config = {'opticon': opticon, 'cfg_id': cfg_id, 'ech_ang': ea, 'pri_ang': pa,
    #                    'n_mats': n_mats, 'mat_order': mat_order, 'w_min': w_min, 'w_max': w_max}
    #     mat_shape = mat_order, mat_order
    #
    #     transform_list = []
    #     tr_hdr_keys = (['ECH_ORDER', 'SLICE_NO', 'SPIFU_NO'])
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
    #
    #         transform_list.append(transform)
    #     hdu_list.close()
    #     return transform_list

    def read_svd_transforms(self, inc_tags=[], exc_tags=[]):
        fits_file_list = Filer.get_file_list(self.tf_dir, inc_tags=inc_tags, exc_tags=exc_tags)
        transform_list = []
        for fits_name in fits_file_list:
            fits_path = self.tf_dir + fits_name
            hdu_list = fits.open(fits_path, mode='readonly')
            primary_hdr = hdu_list[0].header
            pri_cfg = Transform.decode_primary_hdr(primary_hdr)
            n_transforms = len(hdu_list) - 1
            for hdu in hdu_list[1:]:
                transform = Transform(cfg=pri_cfg)
                transform.ingest_extension_hdr(hdu.header)
                transform.read_matrices(hdu.data)
                transform_list.append(transform)
            hdu_list.close()
            # transforms = self.read_svd_transform(fits_name)
            # for transform in transforms:
            #     transform_list.append(transform)
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
