import os
from os import listdir
import pickle
import dill
from astropy.table import Table
from astropy.io import fits
from astropy.io.fits import Card, HDUList, ImageHDU, PrimaryHDU
from lms_globals import Globals
from lms_transform import Transform
import numpy as np


class Filer:

    analysis_types = ['distortion', 'iq']
    test_data_folder = None

    def __init__(self):
        self.model_configuration = None
        self.data_folder, self.sim_folder, self.output_folder = None, None, None
        self.tf_dir, self.trace_file, self.poly_file = None, None, None
        self.wcal_file, self.stats_file, self.tf_fit_file, self.cube_folder = None, None, None, None
        return

    def set_configuration(self, analysis_type, opticon):
        model_configuration = Globals.model_configurations[analysis_type][opticon]
        analysis_type, opticon, date_stamp, _, _, _ = model_configuration
        self.model_configuration = model_configuration
        sub_folder = "{:s}/{:s}/{:s}".format(analysis_type, opticon, date_stamp)
        self.data_folder = Filer.get_folder('../data/model/' + sub_folder)
        self.sim_folder = Filer.get_folder('../data/sim/' + sub_folder)
        self.output_folder = Filer.get_folder('../output/' + sub_folder)
        file_leader = self.output_folder + sub_folder.replace('/', '_')
        self.tf_dir = Filer.get_folder(self.output_folder + 'fits')
        self.trace_file = file_leader + '_trace'  # All ray coordinates
        self.poly_file = file_leader + '_dist_poly.txt'
        self.wcal_file = file_leader + '_dist_wcal.txt'        # Echelle angle as function of wavelength
        self.stats_file = file_leader + '_dist_stats.txt'
        self.tf_fit_file = file_leader + '_dist_tf_fit'        # Use pkl files to write objects directly
        self.cube_folder = Filer.get_folder(self.output_folder + '/cube')
        return

    @staticmethod
    def read_zemax_fits(path):
        """ Read in a Zemax model (PSF) image, with the image data in the primary extension.
        """
        hdu_list = fits.open(path, mode='readonly')
        return hdu_list

    @staticmethod
    def write_zemax_fits(path, header, img_hdu):
        """ Write a Zemax image into the primary extension of a new fits file.
        """
        pri_hdu = PrimaryHDU(header=header, data=img_hdu)
        hdu_list = HDUList([pri_hdu])
        hdu_list.writeto(path, overwrite=True, checksum=True)
        return

    @staticmethod
    def read_mosaic(folder, file_name):
        """ Read in fits file containing LMS detector images.
        :return mosaic tuple (file name, primary extension header, hdu list)
        """
        path = folder + '/' + file_name
        if Globals.is_debug('low'):
            print("Reading {:s}".format(path))
        hdu_in_list = fits.open(path, mode='readonly')
        primary_hdr = hdu_in_list[0].header
        hdu_list = [None]*4       # Re-order hdus
        for hdu_in in hdu_in_list[1:]:
            det_no = int(hdu_in.header['ID'])
            x = hdu_in.header['CRVAL1D']
            y = hdu_in.header['CRVAL2D']
            mos_idx = Globals.mos_idx[det_no]
            if Globals.is_debug('high'):
                fmt = "- storing ScopeSim det_no= {:d} (x, y) = ({:6.3f}, {:6.3f}), at mosaic list index= {:d}"
                print(fmt.format(det_no, x, y, mos_idx))
            hdu_list[mos_idx] = hdu_in
        mosaic = file_name, primary_hdr, hdu_list
        return mosaic

    @staticmethod
    def read_mosaic_list(*args):
        """ Read an LMS data file into a mosaic tuple.  For ScopeSim data, the HDU.header['ID'] holds the detector
        number, ordered det 2 (TR), 1 (TL), 3 (BL), 4 (BR) for extensions 1, 2, 3, 4.   Here, T=Top (slices 15 to 28,
        B = Bottom (slices 1 to 14), L = Left (short wavelength), R = Right (long wavelength).
        We write these into the mosaic tuple as a list, with indices = 0 (TL), 1 (TR), 2 (BL), 3 (BR).
        """
        n_args = len(args)
        inc_tags = args[0]
        exc_tags = args[1] if n_args > 1 else []
        mosaic_list = []
        folder = Filer.test_data_folder
        file_list = Filer.get_file_list(folder, inc_tags=inc_tags, exc_tags=exc_tags)
        if len(file_list) == 0:
            text = "Files in {:s} including tags ".format(folder)
            for tag in inc_tags:
                text += "{:s}, ".format(tag)
            text += 'not found'
            return mosaic_list

        for file_name in file_list:
            mosaic = Filer.read_mosaic(folder, file_name)
            mosaic_list.append(mosaic)
        return mosaic_list

    @staticmethod
    def write_mosaic(folder, mosaic):
        """ Write a Zemax image into the primary extension of a new fits file.
        """
        file_name, primary_header, hdu_list_in = mosaic
        path = folder + '/' + file_name
        primary_hdu = PrimaryHDU(header=primary_header)
        hdu_list = HDUList([primary_hdu])
        for hdu in hdu_list_in:
            hdu_list.append(hdu)
        hdu_list.writeto(path, overwrite=True, checksum=True)
        return

    @staticmethod
    def set_test_data_folder(simulator_name):
        Filer.test_data_folder = "../data/{:s}".format(simulator_name)
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

    @staticmethod
    def read_pinholes(file_name, xy_filter=(0.5, 1.0)):
        path = './inst_pkgs/METIS/wcu/' + file_name + '.dat'
        efp_xy_list = []
        if Globals.is_debug('high'):
            print('Reading pinhole mask from {:s}'.format(file_name))
        with open(path, 'r') as text_file:
            records = text_file.read().splitlines()
            for record in records:
                if Globals.is_debug('high'):
                    print(record)
                if '#' in record:  # Skip comment lines
                    continue
                if 'x' in record:  # Skip column label line
                    continue
                tokens = record.split()
                if len(tokens) < 2: continue
                efp_x = float(tokens[0])
                efp_y = float(tokens[1])
                if (abs(efp_x) < xy_filter[0] and abs(efp_y) < xy_filter[1]):
                    efp_xy_list.append([efp_x, efp_y])
        return efp_xy_list


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
        n_coeffs = wpa_fit['n_coeffs']
        for i in range(0, n_coeffs):
            wpa_col_names.append("WP{:d}".format(i))
        wpa_data = wpa_fit['wpa_opt']     # All slices have the same wxo fit parameters...
        wpa_table = Table(data=np.array(wpa_data), names=wpa_col_names)
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
        print("Reading global fit parameters from file {:s}".format(fits_name))

        wpa_hdr = hdu_list[1].header
        n_wpa_coeffs = wpa_hdr['N_COEFFS']
        wpa_data = hdu_list[1].data
        wpa_fit = {'n_coeffs': n_wpa_coeffs,  'wpa_opt': list(wpa_data[0][:])}

        wxo_header = hdu_list[2].header             # Map from wavelength to echelle angle for slice 13.
        wxo_fit_order, n_wxo_coeffs = wxo_header['ORDER'], wxo_header['N_COEFFS']
        wxo_data = hdu_list[2].data
        slice_no = int(wxo_data['SLICE_NO'][0])
        spifu_no = int(wxo_data['SPIFU_NO'][0])
        wxo_fit = {'order': wxo_fit_order, 'n_coeffs': n_wxo_coeffs,
                   'slice_no': slice_no, 'spifu_no': spifu_no,
                   'wxo_opt': list(wxo_data[0][2:])}

        term_header = hdu_list[3].header
        n_term_coeffs = term_header['N_COEFFS']
        term_data = np.array(hdu_list[3].data)
        svd_order = term_header['ORDER']
        mat_shape = svd_order, svd_order, n_term_coeffs,
        term_fits = {}

        # Should have 18 elements for ext, 28 for nominal
        for data_rec in np.array(term_data):
            data_array = list(data_rec)
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
                mat[mat_row, mat_col] = data_array[data_col: data_col + n_term_coeffs]
                term_fits[slice_no][spifu_no][mat_name] = mat
                data_col += n_term_coeffs
        return wpa_fit, wxo_fit, term_fits

    def write_affine_transform(self, trace):
        _, _, date_stamp, _, _, _ = trace.model_config
        affines = trace.affines
        n_mats, mat_order, _ = affines.shape

        primary_cards = [Card('N_MATS', n_mats, 'MFP <-> DFP transform matrices'),
                         Card('MAT_ORD', mat_order, 'Transform matrix dimensions')
                         ]
        hdr = fits.Header(primary_cards)
        primary_hdu = fits.PrimaryHDU(header=hdr)
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

    def write_svd_transforms(self, trace, **kwargs):
        """ Create HDU binary data tables holding transforms for all slices in a configuration and write them
        to a fits file.  This is the data product we provide to ScopeSim and the pipeline.
        """
        hdu_list = None
        fits_path, fits_name = None, None
        write_primary = True
        for transform in trace.transforms:
            lms_config = transform.lms_configuration
            if write_primary:
                primary_hdu = transform.make_hdu_primary()
                hdu_list = HDUList([primary_hdu])

                # Create fits file with primary HDU only
                otag = '_nom' if lms_config['opticon'] == 'nominal' else '_ext'
                ptag = "_pa{:05d}".format(abs(int(10000. * lms_config['pri_ang'])))
                ea = lms_config['ech_ang']
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

    def read_svd_transforms(self, inc_tags=[], exc_tags=[]):
        """ Read A,B, AI,BI transforms from a fits file.
        """
        fits_file_list = Filer.get_file_list(self.tf_dir, inc_tags=inc_tags, exc_tags=exc_tags)
        transform_list = []
        for fits_name in fits_file_list:
            fits_path = self.tf_dir + fits_name
            hdu_list = fits.open(fits_path, mode='readonly')
            n_ext = len(hdu_list)
            for ext_no in range(1, n_ext):
                transform = Transform(hdu_list=hdu_list, ext_no=ext_no)
                transform_list.append(transform)
            hdu_list.close()
        return transform_list

    # @staticmethod
    # def read_dill(dill_path):
    #     if dill_path[-4:] != '.dil':
    #         dill_path += '.dil'
    #     dill_file = open(dill_path, 'rb')
    #     python_object = dill.load(dill_file)
    #     dill_file.close()
    #     return python_object
    #
    # @staticmethod
    # def write_dill(dill_path, python_object):
    #     dill_file = open(dill_path + '.dil', 'wb')
    #     dill.dump(python_object, dill_file)
    #     dill_file.close()
    #     return

    @staticmethod
    def read_pickle(pickle_path):
        if pickle_path[-4:] != '.pkl':
            pickle_path += '.pkl'
        pickle_file = open(pickle_path, 'rb')
        python_object = pickle.load(pickle_file)
        pickle_file.close()
        return python_object

    @staticmethod
    def write_pickle(pickle_path, python_object):
        pickle_file = open(pickle_path + '.pkl', 'wb')
        pickle.dump(python_object, pickle_file)
        pickle_file.close()
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
