import numpy as np
import time
from lmsiq_fitsio import FitsIo
from lms_wcal import Wcal
from lms_util import Util
from lms_filer import Filer
from lms_detector import Detector
from lms_ipc import Ipc
from lmsiq_plot import Plot
from lmsiq_analyse import Analyse


class Cuber:

    def __init__(self):
        Wcal()
        return

    @staticmethod
    def build(data_identifier, process_control, image_manager, iq_filer):
        """ Build fits cubes (alpha, beta, lambda, configuration, field_position) from the slice images,
        optionally including detector diffusion.
        """
        t_start = time.perf_counter()
        ds_dict, lsf_data = None, None
        axes = ['across-slice', 'along-slice']

        opticon = data_identifier['optical_configuration']
        model_configuration = 'distortion', opticon, '20240109', None, None, None
        dist_filer = Filer(model_configuration)
        traces = dist_filer.read_pickle(dist_filer.trace_file)
        slice_tgt, slice_radius = data_identifier['cube_slice_bounds']

        profile_folder = iq_filer.get_folder(iq_filer.output_folder + 'profiles')

        mc_bounds, inter_pixels = process_control
        im_pix_size = image_manager.model_dict['im_pix_size']
        oversampling = int(Detector.det_pix_size / im_pix_size)

        uni_par = image_manager.unique_parameters
        config_nos = uni_par['config_nos']
        field_nos = uni_par['field_nos']
        spifu_nos = uni_par['spifu_nos']
        slice_nos = uni_par['slice_nos']

        n_runs = mc_bounds[1] - mc_bounds[0] + 3
        n_cube_rows = int(128 / oversampling)
        for ipc_idx, inter_pixel in enumerate(inter_pixels):
            Ipc.set_inter_pixel(inter_pixel)
            ipc_tag = Ipc.tag

            print('\nData set comprises,')
            Util.print_list('field numbers ', field_nos)
            Util.print_list('configurations', config_nos)
            Util.print_list('spifu numbers ', spifu_nos)
            print('cubes will be reconstructed from,')
            Util.print_list('slice numbers ', slice_nos)
            print("(target centred on slice = {:d})".format(slice_tgt))
            print("no. of optical models (2 + Monte-Carlo instances) = {:d}".format(n_runs))

            for field_idx, field_no in enumerate(field_nos):
                field_tag = "field_{:d}".format(field_no)
                slice_start, slice_end = slice_tgt - slice_radius, slice_tgt + slice_radius
                slices_tag = "slices_{:d}_to_{:d}".format(slice_start, slice_end)
                slice_margin = 3
                n_cube_cols = slice_end - slice_start + 1 + 2 * slice_margin

                for spifu_idx, spifu_no in enumerate(spifu_nos):
                    spifu_tag = "spifu_{:d}".format(spifu_no)
                    cube_shape = n_runs, n_cube_rows, n_cube_cols
                    cube = np.zeros(cube_shape)

                    for config_idx, config_no in enumerate(config_nos):
                        config_tag = "cfg_{:d}".format(config_no)
                        fmt = "cube_{:s}_{:s}_{:s}_{:s}_{:s}"
                        cube_name = fmt.format(ipc_tag, field_tag, spifu_tag, config_tag, slices_tag)

                        for slice_no in range(slice_start, slice_end+1):
                            t_now = time.perf_counter()
                            t_min = (t_now - t_start) / 60.
                            fmt = "\r- Diffusion {:s}, field {:d}, spifu_no {:d}, configuration {:d}," + \
                                  " extracting slice_no {:d} at t= {:7.2f} min"
                            print(fmt.format(str(inter_pixel), field_no,
                                             spifu_no, config_no, slice_no, t_min),
                                  end="", flush=True)
                            plot_slicer_images = False
                            if plot_slicer_images:
                                # Don't include IFU image in analysis, just plot it.
                                slicer_obs_list = image_manager.load_dataset(iq_filer,
                                                                             focal_plane='slicer',
                                                                             config_no=config_no,
                                                                             field_no=field_no,
                                                                             mc_bounds=mc_bounds,
                                                                             debug=False
                                                                             )
                                slicer_folder = '/slicer_image'
                                png_name = "slicer_config{:02d}_{:s}".format(config_no, ipc_tag)
                                png_folder = iq_filer.get_folder(iq_filer.output_folder + slicer_folder)
                                png_path = png_folder + png_name
                                title = png_name
                                Plot.images(slicer_obs_list,
                                            figsize=[8, 10], nrowcol=(2, 2), shrink=1.0, colourbar=True,
                                            title=title, do_log=True, png_path=png_path)

                            selection = {'config_no': config_no,
                                         'field_no': field_no,
                                         'focal_plane': 'det',
                                         'slice_no': slice_no,
                                         'spifu_no': spifu_no,
                                         'mc_bounds': mc_bounds,
                                         }
                            images, ds_dict = image_manager.load_dataset(iq_filer,
                                                                         selection,
                                                                         xy_shift=(10, -5, -5),
                                                                         debug=False
                                                                         )

                            # Collect fully processed images for image quality calculation
                            det_images, ipc_images = [], []
                            for zem_img in images:
                                ipc_img = Ipc.apply(zem_img, oversampling) if inter_pixel else zem_img
                                ipc_images.append(ipc_img)
                                det_img = Detector.measure(ipc_img, im_pix_size)
                                det_images.append(det_img)
                            # Find the line widths (perfect, design, mc_mean etc.) for the 'target' slice
                            if slice_no == slice_tgt:

                                # Calculate line spread function parameters
                                axis_idx, axis_name = 0, 'spectral'
                                lsf_data = Analyse.lsf(ipc_images, ds_dict, axis_idx,
                                                       oversample=oversampling,
                                                       debug=False,
                                                       v_coadd=12.0,
                                                       u_radius='all')

                                lsf_folder = Filer.get_folder(profile_folder + 'lsf')
                                lsf_name = 'lsf_' + axis_name + '_' + cube_name
                                lsf_path = lsf_folder + lsf_name
                                write_pickle = False
                                if write_pickle:
                                    iq_filer.write_pickle(lsf_path, (lsf_data, ds_dict))
                                Plot.plot_cube_profile('lsf', lsf_data, lsf_name,
                                                       ds_dict, axis_name,
                                                       xlim=[-6., 6.],
                                                       png_path=lsf_path)
                                dw_dx = Analyse.get_dispersion(ds_dict, traces)  # nm / mm
                                dw_dx_mean, dw_dx_std = dw_dx
                                dw_lmspix = dw_dx_mean * Detector.det_pix_size / 1000.  # nm / pixel
                                # Add FWHM values to summary table (params v wavelength/field etc.)
                                field_series = {'ipc_on': inter_pixel, 'field_no': field_no,
                                                'spifu_no': spifu_no, 'config_no': config_no,
                                                'dw_dlmspix': dw_lmspix,
                                                'gau_fwhms': lsf_data['gau_fwhm'],
                                                'lin_fwhms': lsf_data['lin_fwhm']
                                                }
                                # field_series['wavelength'] = ds_dict['wavelength']
                                for ds_key in ds_dict:
                                    field_series[ds_key] = ds_dict[ds_key]

                            # Add slice to data cube
                            col_idx = slice_no - slice_start + slice_margin
                            det_strips = Analyse.extract_cube_strips(det_images, oversampling)
                            cube[:, :, col_idx] = det_strips

                        plot_images = False
                        if plot_images:
                            pane_titles = ['perfect', 'design', 'MC-001', 'MC-002']
                            png_folder = Filer.get_folder(iq_filer.output_folder + 'cube/png')
                            png_path = png_folder + cube_name
                            plot_image_list = list(cube[0:4])
                            Plot.images(plot_image_list,
                                        aspect='auto',
                                        title=cube_name, png_path=png_path,
                                        pane_titles=pane_titles, nrowcol=(2, 2))

                        for axis, axis_name in enumerate(axes):
                            ee_data = Analyse.eed(cube, ds_dict, axis_name,
                                                  oversample=1,
                                                  debug=False,
                                                  log10sampling=True,
                                                  normalise='to_average')

                            ee_folder = Filer.get_folder(profile_folder + 'ee')
                            ee_name = 'ee_' + axis_name + '_' + cube_name
                            ee_path = ee_folder + ee_name
                            write_pickle = False
                            if write_pickle:
                                iq_filer.write_pickle(ee_path, (ee_data, ds_dict))
                            Plot.plot_cube_profile('ee', ee_data, ee_name, ds_dict, axis_name,
                                                   xlog=True, png_path=ee_path)

                            lsf_folder = Filer.get_folder(profile_folder + 'lsf')
                            lsf_name = 'lsf_' + axis_name + '_' + cube_name
                            lsf_path = lsf_folder + lsf_name
                            lsf_data = Analyse.lsf(cube, ds_dict, axis,
                                                   oversample=1,        # Cube is in detector pixels...
                                                   debug=False,
                                                   v_coadd=3.0,
                                                   u_radius='all')
                            Plot.plot_cube_profile('lsf', lsf_data, lsf_name,
                                                   ds_dict, axis_name,
                                                   png_path=lsf_path)

                        strehls = Analyse.find_strehls(cube)  # Find Strehls for all runs
                        field_series['spifu_no'] = spifu_no
                        field_series['strehls'] = strehls

                        FitsIo.write_cube(cube, cube_name, iq_filer)
                        cube_ident = cube_name, ipc_tag, field_no, spifu_no, config_no
                        cube_package = cube_ident, cube, field_series

                        cube_pkl_folder = iq_filer.get_folder(iq_filer.cube_folder + 'pkl')
                        cube_series_path = cube_pkl_folder + cube_name
                        iq_filer.write_pickle(cube_series_path, cube_package)
        return

    @staticmethod
    def read_pkl_cubes(iq_filer):
        """ Read pkl files in the pkl directory.
        """
        inc_tags, excl_tags = [], []
        cube_pkl_folder = iq_filer.get_folder(iq_filer.cube_folder + 'pkl')
        file_list = iq_filer.get_file_list(cube_pkl_folder,
                                           inc_tags=inc_tags,
                                           excl_tags=['slicer'])
        cube_packages = []
        for file in file_list:
            file_path = cube_pkl_folder + file
            cube_package = iq_filer.read_pickle(file_path)
            cube_packages.append(cube_package)
        return cube_packages

    @staticmethod
    def remove_configs(cube_packages, remove_config_list):
        cube_packages_out = []
        for cube_package in cube_packages:
            cube_ident, cube, field_series = cube_package
            config_no = field_series['config_no']
            if config_no not in remove_config_list:
                cube_packages_out.append(cube_package)
        return cube_packages_out

    # @staticmethod
    # def _field_series_to_csv_lines(field_series, is_first_series):
    #     csv_value_keys = {'prism_angle': (['Prism angle', 'deg.'], '12.4f'),
    #                       'grating_angle': (['Grating angle', 'deg'], '14.4f'),
    #                       'grating_order': (['Grating order', '-'], '14d'),
    #                       'wavelength': (['Wavelength', '-'], '12.6f'),
    #                       'field_no': (['Field no.', '-'], '11d'),
    #                       'spifu_no': (['Sp IFU no.', '-'], '12d')
    #                       }
    #     csv_array_keys = {'strehls': (['<Strehl>', '-'], '10.3f'),
    #                       'gau_fwhms': (['<Gauss FWHM>', 'pix.'], '15.3f'),
    #                       'lin_fwhms': (['<Linear FWHM>', 'pix.'], '15.3f')}
    #     table_list = ['strehls', 'gau_fwhms', 'lin_fwhms']
    #     common_cols = ['prism_angle', 'grating_angle', 'grating_order', 'wavelength']
    #     for key in common_cols:
    #         waves = field_series['wavelength']
    #         csv_text = ''
    #         if is_first_series:
    #             for i in [0, 1]:
    #                 line = ''
    #                 for col in col_list:
    #                     for key in csv_dict:
    #                         tokens, token_fmt = csv_dict[key]
    #                         wid = token_fmt[0:2]
    #                         fmt = '{:' + wid + 's},'
    #                         text = fmt.format(tokens[i])
    #                         line += text
    #                 csv_text += "\n{:s}".format(line)
    #         line = ''
    #         for dict_idx, csv_dict in enumerate([csv_value_keys, csv_array_keys]):
    #             for key in csv_dict:
    #                 val = field_series[key]
    #                 if dict_idx > 0:        # Array data, find the M-C mean
    #                     val = np.mean(val[2:])
    #                 _, token_fmt = csv_dict[key]
    #                 fmt = '{:' + token_fmt + '},'
    #                 text = fmt.format(val)
    #                 line += text
    #         csv_text += "\n{:s}".format(line)
    #     return csv_text

    @staticmethod
    def write_csv(optical_path, cube_packages, iq_filer, to_csv=True):
        csv_value_keys = {'prism_angle': (['Prism angle', 'deg.'], '12.4f'),
                          'grating_angle': (['Grating angle', 'deg'], '14.4f'),
                          'grating_order': (['Grating order', '-'], '14d'),
                          'wavelength': (['Wavelength', '-'], '12.6f'),
                          }
        n_common_cols = len(csv_value_keys)
        csv_array_keys = {'strehls': (['<Strehl>', '-'], '10.3f'),
                          'gau_fwhms': (['<Gauss FWHM>', 'pix.'], '15.3f'),
                          'lin_fwhms': (['<Linear FWHM>', 'pix.'], '15.3f')}

        uni_field_nos, uni_config_nos, uni_spifu_nos = [], [], []
        is_spifu = optical_path == 'spifu'
        # Make array of common values
        for cube_package in cube_packages:
            _, _, field_series = cube_package
            field_no = field_series['field_no']
            if field_no not in uni_field_nos:
                uni_field_nos.append(field_no)
            config_no = field_series['config_no']
            if config_no not in uni_config_nos:
                uni_config_nos.append(config_no)
            spifu_no = field_series['spifu_no']
            if spifu_no not in uni_spifu_nos:
                uni_spifu_nos.append(spifu_no)

        n_rows, n_data_cols = len(uni_config_nos), len(uni_field_nos)
        if is_spifu:
            n_rows = len(uni_spifu_nos)
        blocks = {}
        for key in csv_array_keys:
            blocks[key] = np.zeros((n_rows, n_common_cols + n_data_cols))

        for cube_package in cube_packages:
            _, _, field_series = cube_package
            field_no = field_series['field_no']
            config_no = field_series['config_no']
            spifu_no = field_series['spifu_no']

            r = spifu_no - 1 if is_spifu else config_no - 1  # Row in data block
            for akey in csv_array_keys:
                block = blocks[akey]
                block[r, 0] = field_series['prism_angle']
                block[r, 1] = field_series['grating_angle']
                block[r, 2] = field_series['grating_order']
                block[r, 3] = spifu_no if is_spifu else field_series['wavelength']

                vals = field_series[akey]
                v = np.mean(vals[2:])
                c = n_common_cols + field_no - 1
                block[r, c] = v

        common_fmts = ['{:8.4f},', '{:6.1f},', '{:6.0f},', '{:8.3f},']
        val_fmt = '{:8.3f},'
        csv_text_block = ''
        for akey in csv_array_keys:
            print()
            print("\n{:s}".format(akey))
            csv_text_block += "\n{:s}".format(akey)
            block = blocks[akey]
            # Sort by wavelength
            waves = block[:, 3]
            idx = np.argsort(waves)
            sblock = block[idx, :]
            nr, nc = block.shape
            for r in range(0, nr):
                row = '\n'
                for c in range(0, nc):
                    fmt = common_fmts[c] if c < 4 else val_fmt
                    row += fmt.format(sblock[r, c])
                csv_text_block += row
                print(row)

            csv_folder = Filer.get_folder(iq_filer.output_folder + 'cube/series/csv')
            csv_path = csv_folder + 'series.csv'
            print("Filer.write_profiles to {:s}".format(csv_path))
            with open(csv_path, 'w', newline='') as csv_file:
                print(csv_text_block, file=csv_file)
        return

    @staticmethod
    def plot(optical_path, cube_packages, iq_filer):
        """ Plot all series data.  This method extracts and reformats the plot data from the
        cube packages to make 1 plot per field point.
        """
        print()
        png_folder = Filer.get_folder(iq_filer.output_folder + 'cube/series/png')

        is_spifu = optical_path == 'spifu'

        fig_layouts = {'spifu': (3, 1, (6, 8)), 'nominal': (3, 3, (12, 8))}
        fig_layout = fig_layouts[optical_path]

        key_only = True
        for ipc_on in [True]:
            plot_data = {}
            unique_field_nos = []
            for cube_package in cube_packages:
                _, _, field_series = cube_package
                field_no = field_series['field_no']
                field_id = "field_{:d}".format(field_no)
                if field_no not in unique_field_nos:
                    unique_field_nos.append(field_no)
                    field_data = {'field_no': field_no, 'ipc_on': ipc_on, 'is_spifu': is_spifu,
                                  'x_values': [], 'gau_fwhms': [], 'strehls': [], 'srps': []}
                    plot_data[field_id] = field_data
                field_data = plot_data[field_id]
                wave = field_series['wavelength']
                gau_fwhms = np.array(field_series['gau_fwhms'])
                field_data['gau_fwhms'].append(gau_fwhms)
                strehls = np.array(field_series['strehls'])
                field_data['strehls'].append(strehls)
                dw_dlmspix = field_series['dw_dlmspix']
                srps = 1000. * wave / (gau_fwhms * dw_dlmspix)
                field_data['srps'].append(srps)
                x_val = field_series['spifu_no'] if is_spifu else wave
                field_data['x_values'].append(x_val)

            abscissae = {'spifu': ('spifu_no', 'Spectral IFU slice no.', [0.5, 6.5]),
                         'nominal': ('wavelength', 'Wavelength $\mu$m', [2.7, 5.1])}
            abscissa = abscissae[optical_path]

            fwhm_lims = {'spifu': ([2.5, 2.9]), 'nominal': ([1.5, 3.5])}
            srp_lims = {'spifu': [90000, 120000], 'nominal': [70000, 160000]}
            ordinates = {'gau_fwhms': ('gau_fwhms', 'FWHM / pix.', fwhm_lims[optical_path]),
                         'srps': ('srps', 'SRP ($\lambda / \Delta\lambda$)', srp_lims[optical_path]),
                         'strehls': ('strehls', 'Strehl', [0.4, 1.1])}
            mc_percentiles = [(10, 'dashed', 1.0), (90, 'dashed', 1.0)]

            for ykey in ordinates:
                do_fwhm_rqt = ykey == 'gau_fwhms'
                do_srp_rqt = ykey == 'srps'

                ova_tag = "{:s}_v_{:s}".format(ykey, abscissa[0])
                fmt = "{:s}_{:s}_{:s}"
                ipc_tag = 'ipc_on' if ipc_on else 'ipc_off'
                png_file = fmt.format(optical_path, ova_tag, ipc_tag)
                png_path = png_folder + png_file
                ordinate = ordinates[ykey]
                fmt = "\r- Plotting diffusion {:s}, spectral IFU= {:s}, to file {:s}"
                print(fmt.format(str(ipc_on), str(is_spifu), png_file))
                op_text = 'extended spectral coverage' if is_spifu else 'nominal spectral coverage'
                title = "{:s}\n{:s}, {:s}".format(ova_tag, op_text, ipc_tag)
                Plot.series(plot_data,
                            fig_layout=fig_layout, title=title, plot_key=False,
                            abscissa=abscissa, ordinate=ordinate,
                            mc_percentiles=mc_percentiles,
                            png_path=png_path)

                if key_only:
                    # Explicitly plot a key that can be pasted into the report
                    png_path = png_folder + 'key'
                    Plot.series(plot_data,
                                key_only=key_only,
                                fig_layout=fig_layout, title=title,
                                abscissa=abscissa, ordinate=ordinate,
                                mc_percentiles=mc_percentiles,
                                do_fwhm_rqt=do_fwhm_rqt, do_srp_rqt=do_srp_rqt,
                                png_path=png_path)
                    key_only = False
        return
