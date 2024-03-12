import numpy as np
from lms_globals import Globals
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
    def build(data_identifier, inter_pixels, image_manager, iq_filer):
        """ Build fits cubes (alpha, beta, lambda, configuration, field_position) from the slice images,
        optionally including detector diffusion.
        """
        ds_dict = None
        axes = ['across-slice', 'along-slice']

        model_configuration = 'distortion', data_identifier['optical_configuration'], '20240109', None, None, None
        dist_filer = Filer(model_configuration)
        traces = dist_filer.read_pickle(dist_filer.trace_file)
        slice_tgt, slice_radius = data_identifier['cube_slice_bounds']
        slice_start, slice_end = slice_tgt - slice_radius, slice_tgt + slice_radius

        profile_folder = iq_filer.get_folder(iq_filer.output_folder + 'profiles')

        mc_bounds = image_manager.model_dict['mc_bounds']
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
        slice_margin = 3
        n_cube_cols = slice_end - slice_start + 1 + 2 * slice_margin
        cube_series = None

        for ipc_idx, inter_pixel in enumerate(inter_pixels):
            Ipc.set_inter_pixel(inter_pixel)
            ipc_tag = Ipc.tag

            print()
            Util.print_list('field numbers ', field_nos)
            Util.print_list('spifu numbers ', spifu_nos)
            Util.print_list('configurations', config_nos)
            print('cubes will be reconstructed from,')
            Util.print_list('slice numbers ', slice_nos)
            print("Target centred on slice = {:d}".format(slice_tgt))
            print("Number of models (2 + Monte-Carlo instances) = {:d}".format(n_runs))

            for field_idx, field_no in enumerate(field_nos):

                for spifu_idx, spifu_no in enumerate(spifu_nos):

                    cube_shape = n_runs, n_configs, n_cube_rows, n_cube_cols
                    cube_name = None
                    det_cube = np.zeros(cube_shape)
                    # proc_cube contains the cubes for all images.
                    for config_idx, config_no in enumerate(config_nos):
                        cf_tag = "config_{:d}_field_{:d}_".format(config_no, field_no)
                        slice_tag = "slices_{:d}_to_{:d}".format(slice_start, slice_end)
                        spifu_tag = '' if spifu_no == -1 else "spifu_{:d}".format(spifu_no)
                        fmt = "cube_{:s}_{:s}{:s}{:s}"
                        cube_name = fmt.format(ipc_tag, cf_tag, slice_tag, spifu_tag)
                        fmt = "\r- Diffusion {:s}, field {:d} of {:d}, configuration {:d}, " + \
                              "spifu_no {:d}"
                        print(fmt.format(str(inter_pixel), field_no, n_fields,
                                         config_no, spifu_no),
                              end="", flush=True)

                        for slice_no in range(slice_start, slice_end+1):

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
                                png_name = "slicer_config{:02d}_{:s}".format(config_no, Ipc.tag)
                                png_folder = iq_filer.get_folder(iq_filer.output_folder + slicer_folder)
                                png_path = png_folder + png_name
                                title = png_name
                                Plot.images(slicer_obs_list,
                                            figsize=[8, 10], nrowcol=(2, 2), shrink=1.0, colourbar=True,
                                            title=title, do_log=True, png_path=png_path)

                            pane_titles = ['perfect', 'design', 'MC-001', 'MC-002']
                            images, ds_dict = image_manager.load_dataset(iq_filer,
                                                                         config_no=config_no,
                                                                         field_no=field_no,
                                                                         focal_plane='det',
                                                                         pane_titles=pane_titles,
                                                                         slice_no=slice_no,
                                                                         spifu_no=spifu_no,
                                                                         mc_bounds=mc_bounds,
                                                                         debug=False
                                                                         )

                            # Collect fully processed (un-shifted) images for image quality calculation
                            det_images = []
                            for zem_img in images:
                                ipc_img = Ipc.convolve(zem_img, oversampling) if inter_pixel else zem_img
                                det_img = Detector.measure(ipc_img, im_pix_size)
                                det_images.append(det_img)
                            # Find the line widths (perfect, design, mc_mean etc.) for the 'target' slice
                            if slice_no == slice_tgt:
                                # Create summary table
                                if cube_series is None:
                                    cube_series = {'ipc_on': [], 'field_no': [],
                                                   'spifu_no': [], 'config_no': [],
                                                   'wavelength': [], 'dw_dlmspix': [],
                                                   'fwhms': [], 'strehls': []}
                                    # for key in lsf_data:
                                    #     if 'fwhm' in key:
                                    #         cube_series[key] = []

                                # Calculate line spread function parameters
                                axis, axis_name = 0, 'spectral'
                                lsf_data = Analyse.lsf(det_images, ds_dict, axis,
                                                       oversample=1,
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
                                                       png_path=lsf_path)
                                dw_dx = Analyse.get_dispersion(ds_dict, traces)  # nm / mm
                                dw_dx_mean, dw_dx_std = dw_dx
                                dw_lmspix = dw_dx_mean * Detector.det_pix_size / 1000.  # nm / pixel
                                # Add FWHM values to summary table (params v wavelength/field etc.)
                                cube_series['fwhms'].append(lsf_data['fwhm_lin'])
                                cube_series['ipc_on'].append(inter_pixel)
                                cube_series['field_no'].append(field_no)
                                cube_series['spifu_no'].append(spifu_no)
                                cube_series['config_no'].append(config_no)
                                cube_series['wavelength'].append(ds_dict['wavelength'])
                                cube_series['dw_dlmspix'].append(dw_lmspix)

                                # for key in lsf_data:    # Spectral FWHM on central slice
                                #     if 'fwhm' in key:
                                #         fwhm, _, _ = lsf_data[key]
                                #         cube_series[key].append(fwhm)

                            # Add slice to data cube
                            col_idx = slice_no - slice_start + slice_margin
                            det_strips = Analyse.extract_cube_strips(det_images, oversampling)
                            det_cube[:, config_idx, :, col_idx] = det_strips

                        plot_images = True
                        if plot_images:
                            pane_titles = ['perfect', 'design', 'MC-001', 'MC-002']
                            png_folder = Filer.get_folder(iq_filer.output_folder + 'cube/png')
                            png_path = png_folder + cube_name
                            plot_image_list = list(det_cube[0:4, config_idx])
                            plot_obs_list = plot_image_list, ds_dict
                            Plot.images(plot_obs_list,
                                        title=cube_name, png_path=png_path,
                                        pane_titles=pane_titles, nrowcol=(2, 2))

                        image_list = list(det_cube[:, config_idx])
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

                        strehls = Analyse.find_strehls(image_list)
                        cube_series['strehls'].append(strehls)

                    FitsIo.write_cube(det_cube, cube_name, iq_filer)

        return cube_series

    @staticmethod
    def plot_series(cube_series, iq_filer):

        print()
        field_nos = cube_series['field_no']
        uni_field_nos = np.unique(field_nos)
        n_fields = len(uni_field_nos)
        ipc_on_list = cube_series['ipc_on']
        uni_ipc_states = np.unique(ipc_on_list)
        spifu_nos = cube_series['spifu_no']
        uni_spifu_nos = np.unique(spifu_nos)
        config_nos = cube_series['config_no']
        uni_config_nos = np.unique(config_nos)
        n_spifu_nos = len(uni_spifu_nos)
        # Plot wavelength series data for range of configurations
        for ipc_idx, ipc_on in enumerate(uni_ipc_states):
            Ipc.set_inter_pixel(ipc_on)
            ipc_tag = Ipc.tag

            for field_idx, field_no in enumerate(uni_field_nos):
                for spifu_idx, spifu_no in enumerate(uni_spifu_nos):

                    fmt = "\r- Plotting diffusion {:s}, " + \
                          "field {:d} of {:d}, spifu_no {:d}"
                    print(fmt.format(str(ipc_on),
                                     field_no, n_fields, spifu_no),
                          end="", flush=True)

                    ordinates = [('fwhms', 'pix.'), ('srps', '-')]
                    select = {'ipc_on': ipc_on, 'field_no': field_no, 'spifu_no': spifu_no}
                    png_folder = Filer.get_folder(iq_filer.output_folder + 'cube/series')

                    for ordinate, ordinate_unit in ordinates:
                        fmt = "series_{:s}_{:s}_field_{:02d}_spifu{:02d}"
                        png_file = fmt.format(ordinate, ipc_tag, field_no, spifu_no)
                        fmt = "\r- Plotting diffusion {:s}, " + \
                              "field {:d} of {:d}, spifu_no {:d} to {:s}"
                        print(fmt.format(str(ipc_on), field_no, n_fields, spifu_no, png_file),
                              end="", flush=True)
                        png_path = png_folder + png_file
                        Plot.wav_series(cube_series, select=select,
                                        ordinate=ordinate, ordinate_unit=ordinate_unit,
                                        png_path=png_path)

                    fmt = "series_{:s}_{:s}_field_{:02d}_spifu{:02d}"
                    png_file = fmt.format('strehl', ipc_tag, field_no, spifu_no)
                    png_path = png_folder + png_file
                    Plot.wav_series(cube_series,
                                    ordinate='strehls', ordinate_unit='-',
                                    select=select, png_path=png_path)
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
        # fwhm_mc_gau, fwhm_mc_gau_err = mc_fit[1], mc_fit_err[1]
        _, fwhm_lin_mc, fwhm_lin_mc_err, xl, xr, yh = mc_linear
        line_widths['mc_mean'] = [fwhm_lin_mc, xl, xr]
        return line_widths

    @staticmethod
    def process(config, inter_pixels, iq_filer):
        dataset, n_wavelengths, n_mcruns, slice_locs, folder_name, config_label = config
        waves = []
        for w in range(0, n_wavelengths):
            wave_tag = "{:02d}/".format(w)
            data_folder = dataset + folder_name + wave_tag
            zemax_configuration = FitsIo.read_param_file(dataset, data_folder)
            _, wave, _, _, order, im_pix_size = zemax_configuration
            waves.append(wave)
        for process_level in Globals.process_levels:
            strehl_list = []
            for inter_pixel in inter_pixels:
                Ipc.set_inter_pixel(inter_pixel)
                full_cube, waves, alpha_oversampling = FitsIo.read_cube(process_level, Ipc.tag)
                strehls = Analyse.cube_strehls(full_cube)
                strehl_list.append((strehls, Ipc.tag))

                y_compression = alpha_oversampling * Globals.beta_mas_pix / Globals.alpha_mas_pix
                cube = Analyse.ycompress(full_cube, y_compression)
                n_layers, n_obs, _, _ = cube.shape
                for w in range(0, n_wavelengths):
                    obs_list = []           # Monochromatic images
                    for j in range(0, n_obs):
                        obs_list.append((cube[w, j, :, :], ('', Detector.det_pix_size)))

                    axis = 'radial'
                    oversample = 1.         # After y compression and slice sampling....?
                    eer_data = Analyse.eed(obs_list, axis,
                                           oversample=oversample, debug=False,
                                           log10sampling=True, normalise='to_average')
                    wave = waves[w]
                    plot_cubes = True
                    if plot_cubes:
                        title = "{:s} Reconstructed image at wave = {:6.3f}".format(dataset, wave)
                        parameters = config, 'recon', wave, w, -1, 'ipc'
                        png_folder = iq_filer.iq_png_folder + 'cube'
                        png_folder = iq_filer.get_folder(png_folder)
                        fmt = "cube_{:s}_wav_{:d}_{:s}"
                        png_name = fmt.format(process_level, w, Ipc.tag)
                        png_path = png_folder + png_name
                        fmt = "Cuber.process - Writing image {:s}"
                        print(fmt.format(png_path))
                        Plot.collage(obs_list[0:4], parameters, title=title, png_path=png_path)

                    ee_ref_radius, ee_refs = Analyse.find_ee_axis_references(wave, eer_data)
                    text3 = "{:>12.2f},".format(ee_ref_radius)
                    for key in ee_refs:
                        text3 += "{:>12.6f},".format(ee_refs[key])
                    title = dataset + ', ' + ', Reconstructed Image'
                    plot_cube_profiles = True
                    if plot_cube_profiles:
                        png_path = iq_filer.get_png_path('cube_eer', process_level, w, -1)
                        Plot.plot_ee(eer_data, wave, ee_ref_radius, ee_refs, title, Ipc.tag,
                                     png_path=png_path, plot_all=True)

                    data_id = dataset, '', '', 'proc_zemax', '', '', axis
                    data_type = 'ee_' + 'spatial'
                    fmt = "{:s}cube_ees_{:s}_wav{:d}_{:s}_{:s}.csv"
                    set_path = fmt.format(iq_filer.cube_path, process_level, w, axis, Ipc.tag)
                    iq_filer.write_profiles(data_type, data_id, eer_data, set_path=set_path)
            plot_cube_strehls = True
            if plot_cube_strehls:
                png_folder = iq_filer.iq_png_folder + '/cube_strehls'
                png_folder = iq_filer.get_folder(png_folder)
                png_name = "strehl_{:s}".format(process_level)
                png_path = png_folder + png_name

#                png_path = Filer.get_png_path('cube_strehls', strehl_process_level, -1, -1)
                title = 'Reconstructed {:s} image strehl ratio'.format(process_level)
                Plot.cube_strehls(waves, strehl_list, title, png_path=png_path)
        return
