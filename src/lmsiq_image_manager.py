import numpy as np
from astropy.io import fits


class ImageManager:
    model_dict, config_dict = None, None
    unique_parameters = None
    is_new_zemax = None

    def __init__(self):
        return

    @staticmethod
    def make_dictionary(data_identifier, iq_filer):
        """ Make the image table dictionary which describes all data for this model as a dictionary object
        (config_dict) which has a single entry for each folder/configuration in the model data.  Within a
        folder there are multiple files ('perfect', 'design' and M-C instances) and may be multiple field
        positions, spatial and spectral slices, so these are stored as lists.  A 'dataset' is then defined
        as the unique set of files for a single value of all these parameters,
        eg, folder/config_a, field_b, spat_slice_c, spec_slice_d.
        Method 'load_dataset' then returns the images plus a dictionary (obs_dict) containing the single
        values corresponding to the selected data.
        """
        model_dict = {'optical_path': None,
                      'im_pix_size': None,
                      'fits_trailer': None,
                      'mc_bounds': None,
                      }
        config_dict = {'folders': [],
                       'config_nos': [],
                       'field_nos': [],
                       'focus_shifts': [],
                       'slice_tgts': [],
                       'slice_nos': [], 'spifu_nos': [],
                       'wavelengths': [],
                       'prism_angles': [], 'grating_angles': [], 'grating_orders': [],
                       }
        mc_code, mc_width = 'det', 4
        mc_codelen = len(mc_code)

        dataset_folder = iq_filer.data_folder
        folder_list = iq_filer.get_file_list(dataset_folder, exc_tags=['.csv'])
        folder_codes = {'config': 'config_nos',
                        'field': 'field_nos',
                        'defoc': 'focus_shifts'}

        for folder in folder_list:
            config_dict['folders'].append(folder)
            for code in folder_codes:       # Add folder-wide values
                if code in folder:
                    codelen = len(code)
                    c1 = folder.rfind(code) + codelen
                    cfg_key = folder_codes[code]
                    cfg_tag = folder[c1: c1+3]
                    config_dict[cfg_key].append(int(cfg_tag))

            field_no = config_dict['field_nos'][-1]
            fts = data_identifier['field_tgt_slice']
            slice_tgt = fts[field_no]
            optical_path = data_identifier['optical_path']
            model_dict['optical_path'] = optical_path
            model_dict['im_pix_size'] = 4.5                 # Default image pixel size in microns
            config_dict['slice_tgts'].append(slice_tgt)

            fits_folder = dataset_folder + folder + '/'
            par_files = iq_filer.get_file_list(fits_folder, inc_tags=['.txt'])
            if len(par_files) == 0:
                print("Text file not found in {:s}".format(fits_folder))
            par_path = fits_folder + par_files[0]
            # Note that the slice number information is wrong in the new zemax format (since start of 2024)
            parameters = ImageManager._read_param_file(par_path)

            for key in parameters:
                p = parameters[key]
                if key in model_dict:
                    model_dict[key] = p
                    continue
                cfg_key = key + 's'
                if cfg_key in config_dict:
                    config_dict[cfg_key].append(p)
                    continue
                print("Keyword {:s} not found in model_dict or config_dict".format(key))

            inc_tags = ['.fits']
            exc_tags = ['.txt', 'sli']
            fits_files = iq_filer.get_file_list(fits_folder,
                                                inc_tags=inc_tags,
                                                exc_tags=exc_tags)

            # Read in and decode Monte-Carlo file names and slice/spifu numbers.
            mc_no_list = []
            slice_no_list, spifu_no_list = [], []
            for fits_file in fits_files:
                file_codes = {'spat': ('slice_no', 2, slice_no_list),
                              'spec': ('spifu_no', 1, spifu_no_list)
                              }
                for code in file_codes:
                    if code in fits_file:
                        codelen = len(code)
                        c1 = fits_file.rfind(code) + codelen
                        cfg_key, width, val_list = file_codes[code]
                        val = int(fits_file[c1: c1 + width])
                        if val not in val_list:
                            val_list.append(val)

                if 'sli' in fits_file:
                    continue
                if ('desi' in fits_file) or ('perf' in fits_file):
                    continue
                if mc_code in fits_file:
                    c1 = fits_file.rfind(mc_code) + mc_codelen
                    mc_tag = fits_file[c1: c1 + mc_width]
                    # print(folder, mc_tag)
                    mc_no = int(mc_tag)
                    mc_no_list.append(mc_no)
            if len(mc_no_list) > 0:             # If 'MC' data is included in dataset
                mc_array = np.array(mc_no_list)
                mc_start, mc_end = np.amin(mc_array), np.amax(mc_array)
                model_dict['mc_bounds'] = mc_start, mc_end
            config_dict['slice_nos'].append(slice_no_list)
            config_dict['spifu_nos'].append(spifu_no_list)

        ImageManager._find_unique_parameters(config_dict)
        ImageManager.model_dict = model_dict
        ImageManager.config_dict = config_dict
        return

    # @staticmethod
    # def make_dictionary_old(data_identifier, iq_filer):
    #     """ Make the image table dictionary which describes all data for this model as a dictionary object
    #     (config_dict) which has a single entry for each folder/configuration in the model data.  Within a
    #     folder there are multiple files ('perfect', 'design' and M-C instances) and may be multiple field
    #     positions, spatial and spectral slices, so these are stored as lists.  A 'dataset' is then defined
    #     as the unique set of files for a single value of all these parameters,
    #     eg, folder/config_a, field_b, spat_slice_c, spec_slice_d.
    #     Method 'load_dataset' then returns the images plus a dictionary (obs_dict) containing the single
    #     values corresponding to the selected data.
    #     """
    #     # is_new_zemax = data_identifier['zemax_format'] == 'new_zemax'
    #     # ImageManager.is_new_zemax = is_new_zemax
    #     model_dict = {'optical_path': None,
    #                   'im_pix_size': None,
    #                   'fits_trailer': None,
    #                   'mc_bounds': None,
    #                   }
    #     config_dict = {'folders': [],
    #                    'config_nos': [],
    #                    'field_nos': [],
    #                    'focus_shifts': [],
    #                    'slice_nos': [], 'slice_tags': [], 'slice_tgts': [],
    #                    'spifu_nos': [],
    #                    'wavelengths': [],
    #                    'prism_angles': [], 'grating_angles': [], 'grating_orders': [],
    #                    }
    #     dataset_folder = iq_filer.data_folder
    #     folder_list = iq_filer.get_file_list(dataset_folder, exc_tags=['.csv'])
    #     config_no_list, field_no_list = [], []
    #     for folder in folder_list:
    #         is_valid_field = False
    #         slice_fields = data_identifier['slice_fields']
    #         for key in slice_fields:
    #             field_nos = slice_fields[key]
    #             for field_no in field_nos:
    #                 field_text = '_field{:d}'.format(field_no)
    #                 is_valid_field = True if field_text in folder else is_valid_field
    #         if not is_valid_field:
    #             continue
    #         print('Folder {:s} included in analysis'.format(folder))
    #
    #         optical_path = data_identifier['optical_path']
    #         model_dict['optical_path'] = optical_path
    #         config_dict['folders'].append(folder)
    #         slice_tgt = 13
    #         config_dict['slice_tgts'].append(slice_tgt)
    #
    #         fits_folder = dataset_folder + folder + '/'
    #         par_files = iq_filer.get_file_list(fits_folder, inc_tags=['.txt'])
    #         par_path = fits_folder + par_files[0]
    #         # Note that the slice number information is wrong in the new zemax format (since start of 2024)
    #         parameters = ImageManager._read_param_file(par_path)
    #         for key in parameters:
    #             p = parameters[key]
    #             if key in model_dict:
    #                 model_dict[key] = p
    #                 continue
    #             cfg_key = key + 's'
    #             if cfg_key in config_dict:
    #                 config_dict[cfg_key].append(p)
    #                 continue
    #             print("Keyword {:s} not found in model_dict or config_dict".format(key))
    #
    #         if data_identifier['optical_path'] == 'spifu':
    #             config_dict['prism_angles'].append(6.99716)
    #             config_dict['grating_angles'].append(0.00000)
    #             config_dict['grating_orders'].append(-23)
    #
    #         # Get list of file names for a single run (e.g. identical except for MC run no., 'perfect' or 'design' tags
    #         if optical_path == 'nominal':
    #             exc_tags = ['specslice', 'preslice']
    #             fits_trailer = '_MC_spatslice.fits'
    #         else:
    #             exc_tags = ['spatslice', 'preslice']
    #             fits_trailer = '_MC_specslice.fits'
    #         model_dict['fits_trailer'] = fits_trailer
    #         inc_tags = ['.fits']
    #         fits_files = iq_filer.get_file_list(fits_folder,
    #                                             inc_tags=inc_tags,
    #                                             exc_tags=exc_tags)
    #
    #         tokens = folder.split('_')
    #         field_no = int(tokens[3][-2:])
    #         field_no_list.append(field_no)
    #         config_no = int(tokens[4][-2:])
    #         config_no_list.append(config_no)
    #         if len(tokens) > 4:     # Defocus data present
    #             focus_shift = float(tokens[5][1:4])
    #             config_dict['focus_shifts'].append(int(focus_shift))
    #
    #         # Read in and decode Monte-Carlo file names and slice/spifu numbers.
    #         mc_no_list = []
    #         slice_no_list, slice_tag_list, spifu_no_list = [], [], []
    #         for fits_file in fits_files:
    #
    #             tokens = fits_file.split('_')
    #             slice_idx = 3
    #             spifu_idx = slice_idx + 1
    #             if 'slicer' in fits_file:
    #                 continue
    #             slice_tag = tokens[slice_idx]
    #             slice_no = int(slice_tag)
    #             spifu_no = int(tokens[spifu_idx])
    #             is_new_slice = slice_tag not in slice_tag_list
    #             is_new_spifu = spifu_no not in spifu_no_list
    #             if is_new_slice or is_new_spifu:
    #                 slice_tag_list.append(slice_tag)
    #                 slice_no_list.append(slice_no)
    #                 spifu_no_list.append(spifu_no)
    #
    #             if ('design' in fits_file) or ('perfect' in fits_file):
    #                 continue
    #             if '_MC_' in fits_file:
    #                 mc_idx = spifu_idx + 1
    #                 mc_tag = tokens[mc_idx]
    #
    #                 if mc_tag.isdigit():
    #                     mc_no = int(mc_tag)
    #                     mc_no_list.append(mc_no)
    #         if len(mc_no_list) > 0:             # If 'MC' data is included in dataset
    #             mc_array = np.array(mc_no_list)
    #             mc_start, mc_end = np.amin(mc_array), np.amax(mc_array)
    #             model_dict['mc_bounds'] = mc_start, mc_end
    #         config_dict['slice_tags'].append(slice_tag_list)
    #         config_dict['slice_nos'].append(slice_no_list)
    #         spifu_no_list = [0] if len(spifu_no_list) == 0 else spifu_no_list
    #         config_dict['spifu_nos'].append(spifu_no_list)
    #         config_dict['config_nos'].append(config_no)
    #         config_dict['field_nos'].append(field_no)
    #
    #     ImageManager._find_unique_parameters(config_dict)
    #     ImageManager.model_dict = model_dict
    #     ImageManager.config_dict = config_dict
    #     return

    @staticmethod
    def _find_new_zemax_parameters(config_no, optical_configuration, date_stamp):
        if date_stamp not in ['20240209', '20240229', '20240305']:
            print("ImageManager._find_new_zemax_parameters - NOT VALID FOR {:s}".format(date_stamp))
            return None, None
        spat_slice_no, spec_slice_no = -1, -1
        config_index = config_no - 1
        if optical_configuration == 'nominal':
            spat_slice_no = 12 + config_index % 3
        if optical_configuration == 'spifu':
            spat_slice_no = 12 + config_index % 6
            spec_slice_no = config_index % 3
        return spat_slice_no, spec_slice_no

    @staticmethod
    def _find_unique_parameters(image_table):
        uni_par = {}
        for key in image_table:
            uni_par[key] = []
            values = image_table[key]
            for value in values:
                if key in ['slice_nos', 'spifu_nos']:      # Trap special case of list of lists
                    for val_item in value:
                        if val_item in uni_par[key]:
                            continue
                        uni_par[key].append(val_item)
                    continue
                if value in uni_par[key]:
                    continue
                uni_par[key].append(value)
        ImageManager.unique_parameters = uni_par
        return

    @staticmethod
    def _read_param_file(path):
        params = {}
        pf = open(path, 'r')
        lines = pf.read().splitlines()
        pf.close()
        # Note, field position format changed from [x, y] to single integer in 20240209 dataset
        param_translator = {'wavelength': ('wavelength', 'fl_fl'),
                            'prism angle': ('prism_angle', 'fl_fl'),
                            'grating angle': ('grating_angle', 'fl_fl'),
                            'grating order': ('grating_order', 'fl_int'),
                            'Pixel size (micron)': ('im_pix_size', 'fl_fl')
                            }
        # Set default parameters for SPIFU mode, they don't appear in spifu parameter files.
        params['prism_angle'] = 7.0
        params['grating_angle'] = 0.0
        params['grating_order'] = -99
        for line in lines:
            if ':' in line:
                tokens = line.split(':')
                if tokens[0] not in param_translator:
                    continue
                key_in = tokens[0]
                value = tokens[1]
                key, fmt = param_translator[key_in]

                if key_in == 'field position':
                    if '[' in value:     # Trap change in field position/number format
                        key = 'field_xy'
                        fmt = '2fl_2fl'
                        params['field_no'] = 0
                    else:
                        key = 'field_no'
                        fmt = 'fl_int'
                        params['field_xy'] = [0., 0.]
                if fmt == 'fl_fl':
                    params[key] = float(value)
                    continue
                if fmt == 'fl_int':
                    params[key] = int(value)
                    continue
                if fmt == '2fl_2fl':
                    v1, v2 = value.split(',')
                    v1 = float(v1.replace('[', ''))
                    v2 = float(v2.replace(']', ''))
                    params[key] = float(v1), float(v2)
                if fmt == 'str_str':
                    params[key] = value.strip()
        return params

    # def read_psf_set(self, iq_filer, ech_ord):
    #     iq_date_stamp = '2024073000'
    #     iq_dataset_folder = '../data/iq/nominal/' + iq_date_stamp + '/'
    #     config_no = 41 - ech_ord
    #     iq_config_str = "_config{:03d}".format(config_no)
    #     iq_field_str = "_field{:03d}".format(1)
    #     iq_defoc_str = '_defoc000um'
    #     iq_config_str = 'lms_' + iq_date_stamp + iq_config_str + iq_field_str + iq_defoc_str
    #     # iq_folder = '../data/iq/nominal/' + iq_dataset + '/lms_2024073000_config020_field001_defoc000um/'
    #     iq_folder = iq_dataset_folder + iq_config_str + '/'
    #     amin, vmin, scale, hw_det_psf = None, None, None, None
    #     psf_dict = {}
    #     for slice_no in range(9, 18):
    #         iq_slice_str = "_spat{:02d}".format(slice_no) + '_spec0_detdesi'
    #         iq_filename = iq_config_str + iq_slice_str + '.fits'
    #         iq_path = iq_folder + iq_filename
    #         hdr, psf = iq_filer.read_fits(iq_path)
    #         # print("slice_no={:d}, psf_max={:10.3e}".format(slice_no, np.amax(psf)))
    #         psf_dict[slice_no] = hdr, psf
    #     return psf_dict
    #
    @staticmethod
    def read_zemax_image(path):
        """ Read in a model zemax image """
        hdu_list = fits.open(path, mode='readonly')
        hdu = hdu_list[0]
        image_in = hdu.data
        image = np.array(image_in)  # Make a copy of the raw image (maybe shift it).
        return image

    # @staticmethod
    # def read_mosaic(path):
    #     """ Read in a model zemax image """
    #     hdu_list = fits.open(path, mode='readonly')
    #     hdu = hdu_list[0]
    #     image_in = hdu.data
    #     image = np.array(image_in)  # Make a copy of the raw image (maybe shift it).
    #     return image
    #
    @staticmethod
    def load_dataset(iq_filer, selection, **kwargs):
        """ Load a data (sub-)set (perfect, design plus MC images).
        """
        model_dict = ImageManager.model_dict
        config_dict = ImageManager.config_dict

        debug = kwargs.get('debug', False)
        xy_shift = kwargs.get('xy_shift', None)
        slice_no = selection['slice_no']
        spifu_no = selection['spifu_no']

        # Find the single data set which matches the selection parameters.
        all_indices = {}
        for kw_key in selection:
            indices = []
            for config_key in config_dict:
                if kw_key in config_key:
                    kw_val = selection[kw_key]
                    cfg_vals = config_dict[config_key]
                    for idx, cfg_val in enumerate(cfg_vals):
                        is_list = isinstance(cfg_val, list)
                        if is_list:
                            if kw_val in cfg_val:
                                indices.append(idx)
                        else:
                            if kw_val == cfg_val:
                                indices.append(idx)
            if len(indices) > 0:
                all_indices[kw_key] = indices

        # Find the index that is common to all 'indices' lists
        common_indices = None
        for idx_key in all_indices:
            indices = all_indices[idx_key]
            if common_indices is None:
                common_indices = indices
                continue
            new_cis = []
            for cidx in common_indices:
                if cidx in indices:
                    new_cis.append(cidx)
            common_indices = new_cis

        n_cis = len(common_indices)
        if n_cis != 1:
            if debug:
                print('!! ImageManager.load_dataset - unique data set not found !!')
                for item in selection:
                    print("{:12s}, {:s}".format(item, str(selection[item])))
            return None, None
        dataset_idx = common_indices[0]

        # Generate the dictionary for this dataset ('ds') combining config, model and keyword data
        ds_dict = {}
        for config_key in config_dict:
            ds_key = config_key[:-1]          # Strip final 's' off keyword (slice_nos -> slice_no etc.)
            ds_value = config_dict[config_key][dataset_idx]
            if ds_key == 'slice_no':
                slice_nos = np.array(config_dict['slice_nos'][dataset_idx])
                list_index = np.argwhere(slice_nos == slice_no)[0][0]
                ds_dict[ds_key] = ds_value[list_index]
                continue
            if ds_key == 'spifu_no':
                spifu_nos = np.array(config_dict['spifu_nos'][dataset_idx])
                list_index = np.argwhere(spifu_nos == spifu_no)[0][0]
                ds_dict[ds_key] = ds_value[list_index]
                continue
            ds_dict[ds_key] = ds_value
        for model_key in model_dict:
            ds_dict[model_key] = model_dict[model_key]
        for kw_key in kwargs:
            ds_dict[kw_key] = kwargs[kw_key]

        folder = ds_dict['folder']
        fits_folder = iq_filer.data_folder + folder + '/'

        slice_no = ds_dict['slice_no']
        spifu_no = ds_dict['spifu_no']
        fmt = "{:s}_spat{:02d}_spec{:01d}_det"
        obs_tag = fmt.format(folder, slice_no, spifu_no)

        # Load perfect and design images first, then MC images if requested (mc_bounds is not None)
        mc_tags = ['perf', 'desi']           # Initialise mc tag list for old format
        mc_bounds = selection['mc_bounds']
        if mc_bounds is not None:
            mc_start, mc_end = mc_bounds
            for mc_no in range(mc_start, mc_end + 1):
                mc_tag = "{:04d}".format(mc_no)
                mc_tags.append(mc_tag)

        # Check no. of files in folder against expectation
        n_slices = len(slice_nos)
        n_spifus = len(spifu_nos)
        n_mcs = mc_bounds[1] - mc_bounds[0] + 3
        n_text = 1
        n_expected = n_slices * n_spifus * n_mcs + n_text
        all_files = iq_filer.get_file_list(fits_folder)
        n_files = len(all_files)
        if n_files != n_expected:
            print("Found {:d} files, {:d} expected in folder {:s}".format(n_files, n_expected, folder))

        images, file_names = [], []
        for mc_tag in mc_tags:
            inc_tags = [mc_tag, obs_tag, '.fits']
            file_list = iq_filer.get_file_list(fits_folder,
                                               inc_tags=inc_tags,
                                               exc_tags=['sli'])
            n_files = len(file_list)
            if n_files > 1:
                print("!! multiple files named {:s} in {:s}".format(file_list[0], folder))
            if n_files < 1:
                print("\n!! No files found with {:s}{:s}".format(obs_tag, mc_tag))

            file = file_list[0]
            if debug:
                fmt = "Reading {:s}, mc_tag={:s}"
                print(fmt.format(file, mc_tag))
            path = fits_folder + file
            hdu_list = fits.open(path, mode='readonly')
            hdu = hdu_list[0]
            image_in = hdu.data
            image = np.array(image_in)          # Make a copy of the raw image (maybe shift it).
            if xy_shift is not None:            # Oversample, shift, resample
                sampling, dr, dc = xy_shift
                n_rows, n_cols = image_in.shape
                # Create super-sampled copy of image
                osim = np.zeros((sampling * n_rows, sampling * n_cols))
                for row in range(0, n_rows):
                    for col in range(0, n_cols):
                        r1, c1 = row * sampling, col * sampling
                        r2, c2 = r1 + sampling, c1 + sampling
                        osim[r1:r2, c1:c2] = image[row, col]
                # Apply the integer pixel shift
                osim = np.roll(osim, (dr, dc), axis=(0, 1))
                # Rebin into image
                for row in range(0, n_rows):
                    for col in range(0, n_cols):
                        r1, c1 = row * sampling, col * sampling
                        r2, c2 = r1 + sampling, c1 + sampling
                        image[row, col] = np.mean(osim[r1:r2, c1:c2])

            images.append(image)
            file_names.append(file)

        ds_dict['file_names'] = file_names
        return images, ds_dict

    @staticmethod
    def _make_inc_tags(config_no, slice_no, mc_tag):
        inc_tags = ['.fits']
        fmt = "{:02d}_{:02d}{:s}"
        id_tag = fmt.format(config_no, slice_no, mc_tag)
        if ImageManager.is_new_zemax:
            id_tag = "_{:02d}".format(config_no+1)
        inc_tags.append(id_tag)
        return inc_tags
