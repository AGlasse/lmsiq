import numpy as np
from lms_globals import Globals
from lmsiq_fitsio import FitsIo
from lms_wcal import Wcal
from lms_dist_util import Util
from lms_filer import Filer
from lms_detector import Detector
from lms_ipc import Ipc
from lmsiq_plot import Plot
from lmsiq_analyse import Analyse
from lmsiq_summariser import Summariser


class Cuber:

    def __init__(self):
        Wcal()
        return

    @staticmethod
    def build(data_identifier, inter_pixels, image_manager, iq_filer):
        """ Build fits cubes (alpha, beta, lambda, configuration, field_position) from the slice images,
        optionally including detector diffusion.
        """

        slice_no = None
        w, waves, config_no = None, None, None
        n_obs, centre_col, n_cube_cols = None, None, None
        srp_gau, srp_gau_err = None, None
        axes = Globals.axes

        model_configuration = 'distortion', data_identifier['optical_configuration'], '20240109'
        dist_filer = Filer(model_configuration)
        traces = dist_filer.read_pickle(dist_filer.trace_file)
        slice_tgt, slice_radius = data_identifier['cube_slice_bounds']
        slice_start, slice_end = slice_tgt - slice_radius, slice_tgt + slice_radius

        im_pix_size = image_manager.model_dict['im_pix_size']
        oversampling = int(Detector.det_pix_size / im_pix_size)

        uni_pars = image_manager.unique_parameters
        spifu_nos = uni_pars['spifu_nos']
        config_nos = uni_pars['config_nos']
        field_nos = uni_pars['field_nos']
        mc_bounds = 0, 3

        n_configs = len(config_nos)
        n_fields = len(field_nos)
        n_runs = mc_bounds[1] - mc_bounds[0] + 3
        n_cube_rows = int(128 / oversampling)
        slice_margin = 2
        n_cube_cols = slice_end - slice_start + 1 + 2 * slice_margin

        ipc_idx = 0
        for inter_pixel in inter_pixels:
            Ipc.set_inter_pixel(inter_pixel)
            ipc_tag = Ipc.tag

            fig1_obs_list = []

            field_idx = 0
            for field_no in field_nos:
                spifu_idx = 0
                for spifu_no in spifu_nos:
                    cube_shape = n_runs, n_configs, n_cube_rows, n_cube_cols
                    # proc_cube contains the cubes for all images.
                    det_cube = np.zeros(cube_shape)

                    config_idx = 0
                    for config_no in config_nos:
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
                                data_type = 'slicer_image'
                                png_name = data_type + "_config{:02d}_{:s}".format(config_no, Ipc.tag)
                                png_folder = iq_filer.get_folder(iq_filer.output_folder + data_type)
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

                            fmt = "\r- Diffusion {:s}, configuration {:d}, field {:d} of {:d}, " + \
                                  "spifu_no {:d}, slice_no {:d}"
                            print(fmt.format(str(inter_pixel), config_no,
                                             field_no, n_fields, spifu_no, slice_no),
                                  end="", flush=True)
                            # Collect fully processed (un-shifted) images for image quality calculation
                            det_images = []
                            for zem_img in images:
                                ipc_img = Ipc.convolve(zem_img, oversampling) if inter_pixel else zem_img
                                det_img = Detector.measure(ipc_img, im_pix_size)
                                det_images.append(det_img)

                            col_idx = slice_no - slice_start + slice_margin
                            det_strips = Analyse.extract_cube_strips(det_images, oversampling)
                            det_cube[:, config_idx, :, col_idx] = det_strips

                        cf_tag = "config_{:d}_field_{:d}_".format(config_no, field_no)
                        slice_tag = "slices_{:d}_to_{:d}".format(slice_start, slice_end)
                        spifu_tag = '' if spifu_no == -1 else "spifu_{:d}".format(spifu_no)
                        fmt = "cube_{:s}_{:s}{:s}{:s}"
                        cube_name = fmt.format(ipc_tag, cf_tag, slice_tag, spifu_tag)

                        FitsIo.write_cube(det_cube, cube_name, iq_filer)

                        plot_images = True
                        if plot_images:
                            pane_titles = ['perfect', 'design', 'MC-001', 'MC-002']
                            png_folder = Filer.get_folder(iq_filer.output_folder + 'cube/png')
                            png_path = png_folder + cube_name
                            image_list = list(det_cube[0:4, config_idx])
                            obs_list = image_list, ds_dict
                            Plot.images(obs_list,
                                        title=cube_name,
                                        png_path=png_path, pane_titles=pane_titles, nrowcol=(2, 2))

                        ipc_factor, srp_lin, srp_lin_err = None, None, None
                        result = ''

                        for axis in axes:
                            axis_tag = '_' + axis[0:4]
                            image_list = list(det_cube[:, config_idx])
                            ee_data = Analyse.eed(image_list, ds_dict, axis, oversample=1,
                                                  debug=True, log10sampling=True,
                                                  normalise='to_average')
                            lsf_data = Analyse.lsf(image_list, ds_dict, axis, oversample=1,
                                                   debug=True,
                                                   v_coadd=3.0,
                                                   u_radius='all')
                            # data_id = ds_dict, Ipc.tag, process_level
                            data_type = 'ee_dfp_' + axis
                            iq_filer.write_profiles(data_type, data_id, ee_data)
                            data_type = 'lsf_dfp_' + axis
                            iq_filer.write_profiles(data_type, data_id, lsf_data)

                            line_widths = {}

                            obs_per = pdp[config_no][0]
                            per_gauss, per_linear = Analyse.find_fwhm(obs_per, pdp_oversampling,
                                                                      debug=False)
                            _, fwhm_per_lin, _, xl, xr, yh = per_linear
                            line_widths['perfect'] = [fwhm_per_lin, xl, xr]

                            obs_des = pdp[config_no][1]
                            des_gauss, des_linear = Analyse.find_fwhm(obs_des, pdp_oversampling,
                                                                      debug=False)
                            _, fwhm_des_lin, _, xl, xr, yh = des_linear
                            line_widths['design'] = [fwhm_per_lin, xl, xr]

                            # Find line widths of Monte-Carlo data
                            mc_obs_list = pdp[config_no][2:]
                            mc_gauss, mc_linear = Analyse.find_fwhm_multi(mc_obs_list, pdp_oversampling,
                                                                          debug=False)
                            _, mc_fit, mc_fit_err = mc_gauss
                            fwhm_mc_gau, fwhm_mc_gau_err = mc_fit[1], mc_fit_err[1]
                            _, fwhm_lin_mc, fwhm_lin_mc_err, xl, xr, yh = mc_linear
                            line_widths['mc_mean'] = [fwhm_lin_mc, xl, xr]

                            plot_profiles = True
                            if plot_profiles:
                                png_name = "lsf_{:s}_config_{:d}_{:s}".format(axis, config_no, Ipc.tag)
                                png_folder = iq_filer.output_folder + 'profiles/'
                                png_folder = iq_filer.get_folder(png_folder)
                                png_path = png_folder + png_name
                                dw_lmspix = 1.0
                                Plot.plot_lsf(lsf_data, ds_dict, axis, dw_lmspix, 'lsf',
                                              Ipc.tag, line_widths,
                                              hwlim=6.0, plot_all=True, png_path=png_path)

                            # axis, xlms, ee_per, ee_des, ee_mean, ee_rms, ee_all = ee_data

                            # ee_ref_radius, ee_axis_refs = Analyse.find_ee_axis_references(wave, ee_data)
                            # summary_kv_pairs = []
                            # for key in ee_axis_refs:
                            #     ee_val = ee_axis_refs[key]
                            #     summary_kv_pairs.append((key, ee_val))
                            # result += Summariser.get_value_text(summary_kv_pairs)
                            plot_profiles = True
                            if plot_profiles:
                                png_name = "ees_{:s}_config_{:d}_{:s}".format(axis, config_no, Ipc.tag)
                                png_folder = iq_filer.output_folder + 'profiles/'
                                png_folder = iq_filer.get_folder(png_folder)
                                png_path = png_folder + png_name
                                Plot.plot_ee(ee_data, ds_dict, axis, 'ees', Ipc.tag,
                                             png_path=png_path, plot_all=True)

                            fwhm_tag = 'fwhm' + axis_tag
                            summary_kv_pairs = [(fwhm_tag + '_lin_mc', fwhm_lin_mc),
                                                (fwhm_tag + '_lin_mc_err', fwhm_lin_mc_err),
                                                (fwhm_tag + '_gau_mc', fwhm_mc_gau),
                                                (fwhm_tag + '_gau_mc_err', fwhm_mc_gau_err),
                                                (fwhm_tag + '_lin_per', fwhm_per_lin),
                                                (fwhm_tag + '_lin_des', fwhm_des_lin),
                                                ]
                            result += Summariser.get_value_text(summary_kv_pairs)
                            wfwhm_lin = dw_lmspix * fwhm_lin_mc
                            wfwhm_gau = dw_lmspix * fwhm_mc_gau
                            if axis == 'spectral':
                                srp_lin = wave / wfwhm_lin
                                srp_lin_err = srp_lin * fwhm_lin_mc_err / fwhm_lin_mc
                                srp_gau = wave / wfwhm_gau
                                srp_gau_err = srp_gau * fwhm_mc_gau_err / fwhm_mc_gau

                            # summary_kv_pairs = [('wave', wave), ('order', order),
                            #                     ('srp_mc_lin', srp_lin), ('srp_mc_lin_err', srp_lin_err),
                            #                     ('srp_mc_gau', srp_gau), ('srp_mc_gau_err', srp_gau_err)]
                            # pre_result = Summariser.get_value_text(summary_kv_pairs)
                            # result = pre_result + result
                            # summary.append(result)

                            strips = Analyse.extract_cube_strips(pdp[config_no], pdp_oversampling)
                            cube[config_no, :, :, strip_col] = strips
                            fmt = "Wave={:d}/{:d}, Strip_col={:d}/{:d} -> {:s}"
                            print(fmt.format(config_no + 1, n_datasets, strip_col + 1, n_cube_cols, process_level))

                        proc_cube[process_level] = cube
                        # for i in range(len(summary)):
                        #     print(summary[i])
                        # Summariser.write_summary(process_level, slice_subfolder, summary, iq_filer)

                for process_level in proc_cube:
                    im_oversampling = oversampling
                    cube = proc_cube[process_level]
                    FitsIo.write_cube(process_level, Ipc.tag, waves, im_oversampling, cube, iq_filer)

            ipc_idx += 1
            config_idx += 1
            field_idx += 1
            spifu_idx += 1

            png_folder = iq_filer.output_folder + 'figures'
        png_folder = iq_filer.get_folder(png_folder)
        png_file = "raw_zemax1_wave{:03d}_slice{:02d}".format(config_no, slice_no)
        png_path = png_folder + png_file
        Plot.images(fig1_obs_list[0:4],
                    figsize=[8, 10], nrowcol=(4, 1), shrink=1.0, colourbar=False,
                    title='Fig 1a', do_log=True, png_path=png_path)
        png_file = "raw_zemax2_wave{:03d}_slice{:02d}".format(config_no, slice_no)
        png_path = png_folder + png_file
        Plot.images(fig1_obs_list[4:5],
                    figsize=[8, 10], nrowcol=(4, 1), shrink=1.0, colourbar=False,
                    title='Fig 1b', do_log=True, png_path=png_path)
        png_file = "raw_zemax3_wave{:03d}_slice{:02d}".format(config_no, slice_no)
        png_path = png_folder + png_file
        Plot.images(fig1_obs_list[5:9],
                    figsize=[8, 10], nrowcol=(4, 1), shrink=1.0, colourbar=False,
                    title='Fig 1c', do_log=True, png_path=png_path)
        print()
        return

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
