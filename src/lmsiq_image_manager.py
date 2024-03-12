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
        is_new_zemax = data_identifier['zemax_format'] == 'new_zemax'
        is_nominal = data_identifier['optical_configuration'] == 'nominal'
        ImageManager.is_new_zemax = is_new_zemax
        model_dict = {'optical_configuration': None,
                      'im_pix_size': None,
                      'fits_trailer': None,
                      'mc_bounds': None,
                      }
        config_dict = {'folders': [], 'fits_leaders': [],
                       'config_nos': [],
                       'field_nos': [],
                       'slice_nos': [], 'slice_tags': [], 'slice_tgts': [],
                       'spifu_nos': [],
                       'wavelengths': [], 'focal_planes': [],
                       'prism_angles': [], 'grating_angles': [], 'grating_orders': [],
                       }
        dataset_folder = iq_filer.data_folder
        folder_list = iq_filer.get_file_list(dataset_folder, exc_tags=['SpecIFU_config.csv'])
        is_new_zemax_nominal = is_new_zemax and is_nominal
        specifu_config = iq_filer.read_specifu_config() if is_new_zemax_nominal else None
        for folder in folder_list:
            optical_configuration = data_identifier['optical_configuration']
            model_dict['optical_configuration'] = optical_configuration
            config_dict['folders'].append(folder)
            slice_tgt = 13
            config_dict['slice_tgts'].append(slice_tgt)

            zip_sub_folder = dataset_folder + folder + '/'
            par_files = iq_filer.get_file_list(zip_sub_folder, inc_tags=['.txt'])
            par_path = zip_sub_folder + par_files[0]
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

            # Get list of file names for a single run (e.g. identical except for MC run no., 'perfect' or 'design' tags
            exc_tags = []
            fits_trailer = '.fits'
            if is_new_zemax:
                if optical_configuration == 'nominal':
                    exc_tags = ['specslice', 'preslice']
                    fits_trailer = '_MC_spatslice.fits'
                else:
                    exc_tags = ['spatslice', 'preslice']
                    fits_trailer = '_MC_specslice.fits'
            inc_tags = [fits_trailer]
            model_dict['fits_trailer'] = fits_trailer
            fits_files = iq_filer.get_file_list(zip_sub_folder,
                                                inc_tags=inc_tags,
                                                exc_tags=exc_tags)

            lead = parameters['fits_leader']    # Get configuration and field number
            c_end = lead.find('field')
            config_no_txt = lead[6:c_end]
            config_no = int(config_no_txt)
            f_end = lead.find('wave')
            n_field_chars = len('field')
            field_no_txt = lead[c_end + n_field_chars:f_end]
            field_no = int(field_no_txt)

            # Read in and decode Monte-Carlo file names and slice/spifu numbers.
            focal_plane = ''
            mc_no_list = []
            slice_no_list, slice_tag_list, spifu_no_list = [], [], []
            for fits_file in fits_files:

                tokens = fits_file.split('_')
                idx = 3 if is_new_zemax else 4
                slice_tag: str = tokens[idx]            # New zemax defaults
                focal_plane = 'det'
                slice_no, spifu_no = None, None
                if slice_tag == 'slicer':
                    focal_plane = slice_tag
                else:
                    date_stamp = data_identifier['iq_date_stamp']
                    if date_stamp in ['20240209']:
                        optical_configuration = data_identifier['optical_configuration']
                        slice_no, spifu_no = ImageManager._find_new_zemax_parameters(config_no,
                                                                                     optical_configuration,
                                                                                     date_stamp)
                    if date_stamp in ['20240305']:
                        config_base = int(int((config_no - 1) / 3.) * 3. + 1.)
                        c1 = int(slice_tag) - config_base
                        c2 = c1 + slice_tgt - 1
                        slice_no = c2
                        spifu_no = -1
                        # print(c1, c2, config_no, slice_tag, slice_no)
                        # config_dict['slice_tags'].append(slice_tag)

                    if slice_no is None:
                        slice_no = int(slice_tag)
                        spifu_no = -1

                is_new_slice = slice_tag not in slice_tag_list
                is_new_spifu = spifu_no not in spifu_no_list
                if is_new_slice or is_new_spifu:
                    slice_tag_list.append(slice_tag)
                    slice_no_list.append(slice_no)
                    spifu_no_list.append(spifu_no)
                mc_token = tokens[idx + 1]
                mc_text = mc_token[0:4]
                if mc_text.isdigit():
                    mc_no = int(mc_text)
                    mc_no_list.append(mc_no)
            mc_array = np.array(mc_no_list)
            mc_start, mc_end = np.amin(mc_array), np.amax(mc_array)
            model_dict['mc_bounds'] = mc_start, mc_end
            config_dict['slice_nos'].append(slice_no_list)
            config_dict['slice_tags'].append(slice_tag_list)
            config_dict['focal_planes'].append(focal_plane)
            if specifu_config is None:      # Prism angle etc. set in parameter file
                config_dict['spifu_nos'].append(-1)
                config_no = int(folder[-2:])
            else:
                configs = specifu_config['Conf']
                idx = configs.index(config_no)
                config_dict['prism_angles'].append(specifu_config['Prism'][idx])
                config_dict['grating_angles'].append(specifu_config['Grism'][idx])
                config_dict['grating_orders'].append(specifu_config['Order'][idx])
            config_dict['slice_nos'].append(slice_no_list)
            config_dict['spifu_nos'].append(spifu_no_list)
            config_dict['config_nos'].append(config_no)
            config_dict['field_nos'].append(field_no)

        ImageManager._find_unique_parameters(config_dict)
        ImageManager.model_dict = model_dict
        ImageManager.config_dict = config_dict
        return

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
        param_translator = {'name': ('fits_leader', 'str_str'),
                            'wavelength': ('wavelength', 'fl_fl'),
                            'prism angle': ('prism_angle', 'fl_fl'), 'grating angle': ('grating_angle', 'fl_fl'),
                            'grating order': ('grating_order', 'fl_int'),
                            'Pixel size (micron)': ('im_pix_size', 'fl_fl')
                            }
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

    @staticmethod
    def load_dataset(iq_filer, **kwargs):
        """ Load a data (sub-)set (perfect, design plus MC images).  Filter = None to read all
        data in class.
        """
        model_dict = ImageManager.model_dict
        config_dict = ImageManager.config_dict
        is_new_zemax = ImageManager.is_new_zemax

        debug = kwargs.get('debug', False)
        slice_no = kwargs.get('slice_no', 13)
        spifu_no = kwargs.get('spigu_no', -1)

        # Find the single data set which matches a config_no, field_no, slice_no, spifu_no
        all_indices = []
        for kw_key in kwargs:
            indices = []
            for config_key in config_dict:
                if kw_key in config_key:
                    kw_val = kwargs[kw_key]
                    cfg_vals = config_dict[config_key]
                    for idx, cfg_val in enumerate(cfg_vals):
                        if kw_val == cfg_val:
                            indices.append(idx)
            if len(indices) > 0:
                all_indices.append(indices)

        # Find the index that is common to all 'indices' lists
        dataset_idx = -1
        for idx in all_indices[0]:
            is_common = True
            for indices in all_indices[1:]:
                is_common = is_common if idx in indices else False
            if is_common:
                dataset_idx = idx
                break
        if dataset_idx < 0:
            print('!! ImageManager.load_dataset - unique data set not found !!')

        # Generate the dictionary for this dataset ('ds') combining config, model and keyword data
        ds_dict = {}
        list_index = np.argwhere(np.array(config_dict['slice_nos'][dataset_idx]) == slice_no)[0][0]
        for config_key in config_dict:
            ds_key = config_key[:-1]          # Strip final 's' off keyword (slice_nos -> slice_no etc.)
            ds_value = config_dict[config_key][dataset_idx]
            if isinstance(ds_value, list):
                # print(ds_value)
                ds_dict[ds_key] = ds_value[list_index]
            else:
                ds_dict[ds_key] = ds_value
        for model_key in model_dict:
            ds_dict[model_key] = model_dict[model_key]
        for kw_key in kwargs:
            ds_dict[kw_key] = kwargs[kw_key]

        focal_plane = ds_dict['focal_plane']
        config_no = ds_dict['config_no']

        slice_tag = ds_dict['slice_tag'] if focal_plane == 'det' else ''
        folder = ds_dict['folder']
        fits_folder = iq_filer.data_folder + folder + '/'

        mc_start, mc_end = ds_dict['mc_bounds']

        # Create list of text filters for reading in a specific model
        fits_leader = ds_dict['fits_leader']
        if focal_plane == 'slicer':
            config_tag = '_slicer'
        else:
            config_tag = "{:s}_{:s}".format(fits_leader, slice_tag)

        # Load perfect and design images first
        mc_tags = ['_perfect', '_design']           # Initialise mc tag list for old format
        if is_new_zemax:
            mc_tags = ['perf', 'design']
        for mc_no in range(mc_start, mc_end + 1):
            mc_tag = "_{:04d}".format(mc_no)
            if is_new_zemax:
                mc_tag += '_MC'
            mc_tags.append(mc_tag)

        images, file_names = [], []
        for mc_tag in mc_tags:
            inc_tags = [mc_tag, config_tag]
            file_list = iq_filer.get_file_list(fits_folder,
                                               inc_tags=inc_tags,
                                               excl_tags=['slicer'])
            n_files = len(file_list)
            if n_files > 1:
                print("!! multiple files named {:s} in {:s}".format(file_list[0], folder))
            if n_files < 1:
                print("!! No files found with {:s} and {:s}".format(mc_tag, config_tag))

            file = file_list[0]
            if debug:
                fmt = "Reading {:s}, sno_tag={:s}, mc_tag={:s}"
                print(fmt.format(file, slice_tag, mc_tag))
            path = fits_folder + file
            hdu_list = fits.open(path, mode='readonly')
            hdu = hdu_list[0]
            images.append(hdu.data)
            file_names.append(file)
        ds_dict['file_names'] = file_names
        return images, ds_dict

    @staticmethod
    def _make_inc_tags(config_no, slice_no, mc_tag):
        # field_no = kwargs.get('field_no', None)
        # wave_no = kwargs.get('wave_no', None)
        inc_tags = ['.fits']
        fmt = "{:02d}_{:02d}{:s}"
        id_tag = fmt.format(config_no, slice_no, mc_tag)
        if ImageManager.is_new_zemax:
            id_tag = "_{:02d}".format(config_no+1)
        inc_tags.append(id_tag)
        return inc_tags
