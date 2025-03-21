import os
from os import listdir
import pickle
from astropy.table import Table
from astropy.io import fits
from astropy.io.fits import Card, HDUList, ImageHDU, PrimaryHDU
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
    def read_fits(path, header_ext=0, data_exts=[0]):
        """ Read in a model zemax image """
        hdu_list = fits.open(path, mode='readonly')
        header = hdu_list[header_ext].header
        image_list = [hdu_list[i].data for i in data_exts]
        if len(data_exts) == 1:
            return header, image_list[0]
        return header, image_list

    @staticmethod
    def write_fits(path, header, data):
        """ Write Zemax image.  Header in hdu[0].  If data is a (3D) image stack, they are written to hdu[1] onwards,
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

    def write_fit_parameters(self, wxo_fit, term_fits):

        n_slices = len(term_fits)
        primary_cards = [Card('N_SLICES', n_slices, 'No. of slices')]
        primary_header = fits.Header(primary_cards)
        primary_hdu = fits.PrimaryHDU(header=primary_header)
        hdu_list = HDUList([primary_hdu])

        _, opticon, date_stamp, _, _, _ = self.model_configuration

        fmt = "lms_dist_efp_mfp_{:s}_fit_parameters_v{:s}"
        fits_name = fmt.format(opticon[0:3], date_stamp)
        fits_path = self.tf_dir + fits_name + '.fits'

        wxo_column_names = ['SLICE_NO', 'CONST', 'PRI', 'ECH', 'PRIxPRI', 'ECHxECH', 'PRIxECH']
        n_columns = len(wxo_column_names)
        wxo_data = np.zeros((n_columns))
        # Write wavelength x echelle order fit parameters to second HDU
        wxo_data[0] = wxo_fit['slice_no']
        wxo_data[1:n_columns] = wxo_fit['wxo_opt']     # All slices have the same wxo fit parameters...
        wxo_cards = [Card('DESCR', 'Fit parms to map prism and echelle angle to slice 13 wavelength.', '')]
        wxo_hdr = fits.Header(wxo_cards)
        wxo_table = Table(data=wxo_data, names=wxo_column_names)
        wxo_hdu = fits.BinTableHDU(data=wxo_table, header=wxo_hdr)
        hdu_list.append(wxo_hdu)

        term_cards = [Card('DESCR', 'Fit parms to map prism and echelle angle to transform terms.', '')]
        term_hdr = fits.Header(term_cards)
        mat_tags_uc = ['A', 'B', 'AI', 'BI']
        term_names = ['SLICE_NO', 'ROW', 'COL']
        for mat_tag in mat_tags_uc:
            for tag in wxo_column_names[1:7]:
                term_names.append(mat_tag + '_' + tag)
        n_columns = len(term_names)
        n_records_slice = 4 * 4
        n_vals_fit = 6
        term_data = np.zeros((n_slices * n_records_slice, n_columns))
        i = 0                           # Row counter in term_data array
        for slice_idx in range(0, n_slices):
            slice_no = slice_idx + 1
            term_data[i:i + n_records_slice, 0] = slice_no
            mat_fits = term_fits[slice_no]
            for row in range(0, 4):
                for col in range(0, 4):
                    term_data[i, 1:3] = [row, col]
                    j = 3
                    for mat_tag_uc in mat_tags_uc:
                        mat_tag_lc = mat_tag_uc.lower()
                        mat = np.array(mat_fits[mat_tag_lc])
                        term_data[i, j:j + n_vals_fit] = mat[row, col, :]
                        j += n_vals_fit
                    i += 1
        term_table = Table(data=term_data, names=term_names)
        term_hdu = fits.BinTableHDU(data=term_table, header=term_hdr)
        hdu_list.append(term_hdu)
        hdu_list.writeto(fits_path, overwrite=True)
        return

    def read_fit_parameters(self):
        """ Method to read the surface fit parameters from the fits file. """
        _, opticon, date_stamp, _, _, _ = self.model_configuration
        n_slices = 28 if opticon == Globals.nominal else 3

        fmt = "lms_dist_efp_mfp_{:s}_fit_parameters_v{:s}"
        fits_name = fmt.format(opticon[0:3], date_stamp)
        fits_path = self.tf_dir + fits_name + '.fits'
        hdu_list = fits.open(fits_path, mode='readonly')

        wxo_data = hdu_list[1].data
        slice_no = int(wxo_data['SLICE_NO'][0])
        wxo_fit = {'slice_no': slice_no, 'wxo_opt': list(wxo_data[0][1:])}

        term_data = hdu_list[2].data
        mat_length = Globals.svd_order * Globals.svd_order
        mat_shape = Globals.svd_shape + (Globals.n_svd_fit_terms,)
        term_fits = {}
        data_row = 0       # Row counter in term data records
        for slice_idx in range(0, n_slices):
            slice_no = slice_idx + 1
            term_fits[slice_no] = {}
            for mat_name in Globals.matrix_names:
                term_fits[slice_no][mat_name] = np.zeros(mat_shape)
            for i in range(data_row, data_row + mat_length):
                mat_row = int(term_data['ROW'][i])
                mat_col = int(term_data['COL'][i])
                data_col = 3  # Column counter in term data record
                for mat_name in Globals.matrix_names:
                    mat = term_fits[slice_no][mat_name]
                    mat[mat_row, mat_col] = list(term_data[i][data_col:data_col+6])
                    data_col += 6
            data_row += 16
        return wxo_fit, term_fits

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

    def write_svd_transform(self, trace):
        """ Create data tables holding transforms for all slices in a configuration and write them
        to a fits file.
        """
        cfg_id = trace.cfg_id
        pa = trace.parameter['Prism angle']
        ea = trace.parameter['Echelle angle']
        _, opticon, date_stamp, _, _, _ = trace.model_config
        n_mats = Globals.n_svd_matrices
        svd_order = Globals.svd_order

        primary_cards = [Card('OPTICON', opticon, 'Optical configuration'),
                         Card('CFG_ID', cfg_id, 'Optical configuration id'),
                         Card('PRI_ANG', pa, 'Prism angle / deg'),
                         Card('ECH_ANG', ea, 'Echelle angle / deg'),
                         Card('N_MATS', n_mats, 'A, B, AI, BI transform matrices'),
                         Card('MAT_ORD', svd_order, 'Transform matrix dimensions')
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
                     Card('SPIFU', spifu_no, 'Spectral IFU slice no. (=0 if not selected'),
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

    def read_svd_transform(self, fits_name):
        """ Read a list of transforms between the EFP and MFP from a fits file.
        """
        fits_path = self.tf_dir + fits_name
        hdu_list = fits.open(fits_path, mode='readonly')
        primary_hdr = hdu_list[0].header
        opticon = primary_hdr['OPTICON']
        cfg_id = primary_hdr['CFG_ID']
        ea = primary_hdr['ECH_ANG']
        pa = primary_hdr['PRI_ANG']
        n_mats = primary_hdr['N_MATS']
        mat_order = primary_hdr['MAT_ORD']

        base_config = {'opticon': opticon, 'cfg_id': cfg_id, 'ech_ang': ea, 'pri_ang': pa,
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

    def read_svd_transforms(self, inc_tags=[], exc_tags=[]):
        fits_file_list = Filer.get_file_list(self.tf_dir, inc_tags=inc_tags, exc_tags=exc_tags)
        transform_list = []
        for fits_name in fits_file_list:
            transform = self.read_svd_transform(fits_name)
            # for transform in transforms:
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
