import numpy as np
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
        ds_dict, lsf_data = None, None
        axes = ['across-slice', 'along-slice']

        opticon = data_identifier['optical_configuration']
        model_configuration = 'distortion', opticon, '20240109', None, None, None
        dist_filer = Filer(model_configuration)
        traces = dist_filer.read_pickle(dist_filer.trace_file)
        slice_tgt, slice_radius = data_identifier['cube_slice_bounds']

        profile_folder = iq_filer.get_folder(iq_filer.output_folder + 'profiles')

        mc_bounds, inter_pixels = process_control
        # mc_bounds = image_manager.model_dict['mc_bounds']
        im_pix_size = image_manager.model_dict['im_pix_size']
        oversampling = int(Detector.det_pix_size / im_pix_size)

        uni_par = image_manager.unique_parameters
        config_nos = uni_par['config_nos']
        field_nos = uni_par['field_nos']
        slice_nos = uni_par['slice_nos']
        spifu_nos = uni_par['spifu_nos']

        n_configs = len(config_nos)
        n_fields = len(field_nos)
        n_runs = mc_bounds[1] - mc_bounds[0] + 3
        n_cube_rows = int(128 / oversampling)
        for ipc_idx, inter_pixel in enumerate(inter_pixels):
            Ipc.set_inter_pixel(inter_pixel)
            ipc_tag = Ipc.tag

            print('Data set comprises,')
            Util.print_list('field numbers ', field_nos)
            Util.print_list('configurations', config_nos)
            if opticon == 'spifu':
                Util.print_list('spifu numbers ', spifu_nos)
            print('cubes will be reconstructed from,')
            Util.print_list('slice numbers ', slice_nos)
            print("(target centred on slice = {:d})".format(slice_tgt))
            print("no. of optical models (2 + Monte-Carlo instances) = {:d}".format(n_runs))

            for field_idx, field_no in enumerate(field_nos):
                slice_start, slice_end = slice_tgt - slice_radius, slice_tgt + slice_radius
                slices_tag = "slices_{:d}_to_{:d}".format(slice_start, slice_end)
                slice_margin = 3
                n_cube_cols = slice_end - slice_start + 1 + 2 * slice_margin

                field_tag = "field_{:d}".format(field_no)
                field_series, field_cubes = None, []  # Data for specific field point to write to pickle file
                for spifu_idx, spifu_no in enumerate(spifu_nos):
                    spifu_tag = '' if spifu_no == 0 else "spifu_{:d}".format(spifu_no)
                    cube_shape = n_runs, n_configs, n_cube_rows, n_cube_cols
                    fits_name = "cube_{:s}_{:s}_{:s}_{:s}".format(ipc_tag, field_tag, spifu_tag, slices_tag)
                    cube_name = None
                    cube_data = np.zeros(cube_shape)
                    # proc_cube contains the cubes for all images.
                    for config_idx, config_no in enumerate(config_nos):
                        config_tag = "config_{:d}_".format(config_no)
                        fmt = "cube_{:s}_{:s}_{:s}_{:s}_{:s}"
                        cube_name = fmt.format(ipc_tag, field_tag, config_tag, spifu_tag, slices_tag)

                        for slice_no in range(slice_start, slice_end+1):
                            fmt = "\r- Diffusion {:s}, field {:d}, configuration {:d}, " + \
                                  "spifu_no {:d}, extracting slice_no {:d}"
                            print(fmt.format(str(inter_pixel), field_no,
                                             config_no, spifu_no, slice_no),
                                  end="", flush=True)
                            plot_slicer_images = False
                            if plot_slicer_images:
                                # Don't include IFU image in analysis, just plot it.
                                slicer_obs_list = image_manager.load_dataset(iq_filer,
                                                                             focal_plane='slicer',
                                                                             # xy_shift=(10, -5, -5),
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

                            # Collect fully processed (un-shifted) images for image quality calculation
                            det_images, ipc_images = [], []
                            for zem_img in images:
                                ipc_img = Ipc.apply(zem_img, oversampling) if inter_pixel else zem_img
                                ipc_images.append(ipc_img)
                                det_img = Detector.measure(ipc_img, im_pix_size)
                                det_images.append(det_img)
                            # Find the line widths (perfect, design, mc_mean etc.) for the 'target' slice
                            if slice_no == slice_tgt:
                                if field_series is None:
                                    field_series = {'ipc_on': [], 'field_no': [],
                                                    'spifu_no': [], 'config_no': [],
                                                    'wavelength': [], 'dw_dlmspix': [],
                                                    'fwhms': [], 'strehls': []}

                                # Calculate line spread function parameters
                                axis, axis_name = 0, 'spectral'
                                lsf_data = Analyse.lsf(ipc_images, ds_dict, axis,
                                                       oversample=oversampling,         # =1 for det images
                                                       boxcar=True,
                                                       debug=False,
                                                       v_coadd=3.0,
                                                       u_radius='all')

                                # lsf_fwhm = Cuber._find_line_widths(det_images, 0)
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
                                field_series['fwhms'].append(lsf_data['fwhm_lin'])
                                field_series['ipc_on'].append(inter_pixel)
                                field_series['field_no'].append(field_no)
                                field_series['spifu_no'].append(spifu_no)
                                field_series['config_no'].append(config_no)
                                field_series['wavelength'].append(ds_dict['wavelength'])
                                field_series['dw_dlmspix'].append(dw_lmspix)

                            # Add slice to data cube
                            col_idx = slice_no - slice_start + slice_margin
                            det_strips = Analyse.extract_cube_strips(det_images, oversampling)
                            cube_data[:, config_idx, :, col_idx] = det_strips

                        plot_images = True
                        if plot_images:
                            pane_titles = ['perfect', 'design', 'MC-001', 'MC-002']
                            png_folder = Filer.get_folder(iq_filer.output_folder + 'cube/png')
                            png_path = png_folder + cube_name
                            plot_image_list = list(cube_data[0:4, config_idx])
                            Plot.images(plot_image_list,
                                        aspect='auto',
                                        title=cube_name, png_path=png_path,
                                        pane_titles=pane_titles, nrowcol=(2, 2))

                        image_list = list(cube_data[:, config_idx])
                        for axis, axis_name in enumerate(axes):
                            ee_data = Analyse.eed(image_list, ds_dict, axis_name, lsf_data,
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
                            lsf_data = Analyse.lsf(image_list, ds_dict, axis,
                                                   oversample=1,
                                                   debug=False,
                                                   v_coadd=3.0,
                                                   u_radius='all')
                            Plot.plot_cube_profile('lsf', lsf_data, lsf_name,
                                                   ds_dict, axis_name,
                                                   png_path=lsf_path)

                        strehls = Analyse.find_strehls(cube_data[:, config_idx, :, :])  # Find Strehls for all runs
                        field_series['strehls'].append(strehls)

                    cube_ident = cube_name, ipc_tag, field_no, spifu_no
                    cube = cube_ident, cube_data
                    field_cubes.append(cube)
                    FitsIo.write_cube(cube_data, fits_name, iq_filer)

                cube_pkl_folder = iq_filer.get_folder(iq_filer.cube_folder + 'pkl')
                cube_pkl_name = "series_{:s}_{:s}_{:s}".format(ipc_tag, field_tag, spifu_tag)
                cube_series_path = cube_pkl_folder + cube_pkl_name
                iq_filer.write_pickle(cube_series_path, (field_series, field_cubes))

        return

    @staticmethod
    def remove_configs(cube_series_in, remove_config_list):
        cube_series = {}
        config_list = cube_series_in['config_no']
        indices = np.full(len(config_list), True)
        for rem_config_no in remove_config_list:
            n_items_to_remove = config_list.count(rem_config_no)
            start = 0
            for item in range(0, n_items_to_remove):
                idx = cube_series_in['config_no'].index(rem_config_no, start)
                indices[idx] = False
                start = idx + 1

            for key in cube_series_in:
                val_list_in = cube_series_in[key]
                val_list = []
                for idx, flag in enumerate(indices):
                    if flag:
                        val_list.append(val_list_in[idx])
                cube_series[key] = val_list
        return cube_series

    @staticmethod
    def plot_series(optical_path, cube_series, iq_filer):

        print()
        is_spifu = optical_path == 'spifu'

        field_nos = cube_series['field_no']
        uni_field_nos = np.unique(field_nos)
        n_fields = len(uni_field_nos)
        ipc_on_list = cube_series['ipc_on']
        uni_ipc_states = np.unique(ipc_on_list)
        spifu_nos = cube_series['spifu_no']
        uni_spifu_nos = np.unique(spifu_nos)
        mc_percentiles = [(10, 'dashed', 1.0), (90, 'dashed', 1.0)]
        # Plot wavelength series data for range of configurations
        for ipc_idx, ipc_on in enumerate(uni_ipc_states):
            Ipc.set_inter_pixel(ipc_on)
            ipc_tag = Ipc.tag

            for field_idx, field_no in enumerate(uni_field_nos):
                for spifu_idx, spifu_no in enumerate(uni_spifu_nos):

                    fmt = "\r- Plotting diffusion {:s}, " + \
                          "field {:d} of {:d}, spifu_no {:d}"
                    print(fmt.format(str(ipc_on), field_no, n_fields, spifu_no),
                          end="", flush=True)

                    ordinates = [('fwhms', 'pix.'), ('srps', '-'), ('strehls', '-')]
                    abscissa = ('spifu_no', '-') if is_spifu else ('wavelength', '$\mu$m')

                    select = {'ipc_on': ipc_on, 'field_no': field_no}
                    png_folder = Filer.get_folder(iq_filer.output_folder + 'cube/series')

                    for ordinate in ordinates:
                        ova_tag = "{:s}_v_{:s}".format(ordinate[0], abscissa[0])
                        fmt = "{:s}_{:s}_{:s}_field_{:02d}_spifu_{:02d}"
                        png_file = fmt.format(optical_path, ova_tag, ipc_tag, field_no, spifu_no)
                        fmt = "\r- Plotting diffusion {:s}, " + \
                              "field {:d} of {:d}, spifu_no {:d} to {:s}"
                        print(fmt.format(str(ipc_on), field_no, n_fields, spifu_no, png_file),
                              end="", flush=True)
                        png_path = png_folder + png_file
                        title = png_file

                        Plot.field_series(cube_series, is_spifu,
                                          title=title,
                                          select=select,
                                          ordinate=ordinate,
                                          abscissa=abscissa,
                                          mc_percentiles=mc_percentiles,
                                          png_path=png_path)

        print()
        return

    @staticmethod
    def _find_line_widths(image_list, axis):
        """ Calculate the line widths for a list of images along a specified axis. """
        line_widths = {}
        per_gauss, per_linear = Analyse.find_fwhm(image_list[0], axis=axis, debug=True)
        _, fwhm_per_lin, _, xl, xr, yh = per_linear
        line_widths['perfect'] = [fwhm_per_lin, xl, xr]

        des_gauss, des_linear = Analyse.find_fwhm(image_list[1], axis=axis, debug=True)
        _, fwhm_des_lin, _, xl, xr, yh = des_linear
        line_widths['design'] = [fwhm_per_lin, xl, xr]

        # Find line widths of Monte-Carlo data
        mc_gauss, mc_linear = Analyse.find_fwhm_multi(image_list[2:], axis=axis, debug=True)
        _, mc_fit, mc_fit_err = mc_gauss
        _, fwhm_lin_mc, fwhm_lin_mc_err, xl, xr, yh = mc_linear
        line_widths['mc_mean'] = [fwhm_lin_mc, xl, xr]
        return line_widths
