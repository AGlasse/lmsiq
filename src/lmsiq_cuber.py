import numpy as np
from lms_globals import Globals
from lmsiq_fitsio import FitsIo
from lms_dist_util import Util
from lms_filer import Filer
from lms_detector import Detector
from lms_wcal import Wcal
from lms_ipc import Ipc
from lmsiq_plot import Plot
from lmsiq_analyse import Analyse
from lmsiq_summariser import Summariser


class Cuber:

    def __init__(self):
        return

    @staticmethod
    def build(config, inter_pixels, raytrace_file):
        w, waves = None, None
        n_obs, centre_col, n_cube_cols = None, None, None
        srp_gau, srp_gau_err = None, None
        axes = Globals.axes

        traces = Filer.read_pickle(Filer.trace_file)

        # poly_file = '../output/distortion/nominal/nominal_dist_poly_old.txt'           # Distortion polynomial
        # poly, ech_bounds = Util.read_polyfits_file(poly_file)   # Use distortion map for finding dispersion
        # wcal = Wcal()
        # wcal.read_poly()

        # Multi-slice, multi-wavelength, multi-Monte Carlo run configurations
        optical_path, date_stamp, n_wavelengths, n_mcruns, slice_locs, folder_name, config_label = config
        n_mcruns_tag = "{:04d}".format(n_mcruns)
        tgt_slice_no, n_slices, slice_idents = Util.parse_slice_locations(slice_locs)

        for inter_pixel in inter_pixels:
            Ipc.set_inter_pixel(inter_pixel)
            raw_zemax_cube, proc_zemax_cube, proc_detector_cube = None, None, None
            proc_cube = {'raw_zemax': raw_zemax_cube,
                         'proc_zemax': proc_zemax_cube,
                         'proc_detector': proc_detector_cube}

            fig1_obs_list = []
            for slice_ident in slice_idents:
                slice_no, slice_subfolder, slice_label = slice_ident

                # Generate wavelength dependent summary file from profiles.
                raw_zemax, proc_zemax, proc_detector = [], [], []
                waves = []
                for wave_no in range(0, n_wavelengths):
                    wave_tag = "{:02d}/".format(wave_no)

                    n_obs = n_mcruns + 2  # Include the 'perfect' and 'design' cases at the start of the obs list
                    data_folder = date_stamp + folder_name + wave_tag
                    # For first dataset, read in parameters from text file and generate IPC and IPG kernels.
                    zemax_configuration = FitsIo.read_param_file(date_stamp, data_folder)
                    _, wave, _, _, order, im_pix_size = zemax_configuration
                    waves.append(wave)
                    slice_folder = data_folder + slice_subfolder
                    obs_list = FitsIo.load_dataset(date_stamp, slice_folder, n_mcruns)

                    # Don't include IFU image in analysis, just plot it.
                    if slice_no == -1:
                        png_folder = Filer.png_path + 'ifu'
                        png_folder = Filer.get_folder(png_folder)
                        png_name = "ifu_{:02d}".format(wave_no)
                        png_path = png_folder + png_name
                        Plot.images(obs_list[0:4],
                                    figsize=[8, 10], nrowcol=(2, 2), shrink=1.0, colourbar=True,
                                    title='IFU Zemax image', do_log=True, png_path=png_path)
                        continue

                    is_fig1_obs = wave_no == 0 and not inter_pixel      # Shortest wavelength, no diffusion
                    if is_fig1_obs:
                        fig1_obs_list.append(obs_list[1])       # Use design case

                    # Collect all observations (raw, shifted, processed, detected) for this configuration in lists
                    raw_zemax_row, proc_zemax_row, proc_detected_row = [], [], []
                    for res_col, obs_1 in enumerate(obs_list):
                        fmt = "\r- Slice {:d} wavelength index {:02d} of {:02d}, Obs {:03d} of {:03d}"
                        print(fmt.format(slice_no, wave_no + 1, n_wavelengths, res_col + 1, n_mcruns + 2),
                              end="", flush=True)
                        # Collect fully processed (un-shifted) images for image quality calculation
                        obs_2 = Ipc.convolve(obs_1) if inter_pixel else obs_1
                        obs_3 = Detector.measure(obs_2)
                        raw_zemax_row.append(obs_1)
                        proc_zemax_row.append(obs_2)
                        proc_detected_row.append(obs_3)
                    raw_zemax.append(raw_zemax_row)
                    proc_zemax.append(proc_zemax_row)
                    proc_detector.append(proc_detected_row)

                if slice_no == -1:      # Don't process IFU images.
                    continue
                proc_obs = {'raw_zemax': raw_zemax,
                            'proc_zemax': proc_zemax,
                            'proc_detector': proc_detector}
                for process_level in Globals.process_levels:
                    pdp_oversampling = Globals.get_im_oversampling(process_level)
                    cube = proc_cube[process_level]
                    pdp = proc_obs[process_level]
                    if cube is None:
                        n_cube_rows = pdp[0][0][0].shape[0]
                        slice_margin = 4
                        n_cube_cols = n_slices + 2 * slice_margin
                        centre_col = int(n_cube_cols / 2)
                        cube = np.zeros((n_wavelengths, n_obs, n_cube_rows, n_cube_cols))
                        proc_cube[process_level] = cube
                        print()
                        print("Creating {:s} data cube".format(process_level))
                    strip_col = centre_col + (slice_no - tgt_slice_no)

                    summary = Summariser.create_summary_header(axes)
                    for wave_no in range(0, n_wavelengths):
                        wave_tag = "{:02d}/".format(wave_no)
                        wave = waves[wave_no]
                        cube = proc_cube[process_level]

                        plot_images = True
                        if plot_images:
                            parameters = config, slice_label, wave, wave_no, slice_no, Ipc.tag
                            fmt = "cube_{:s}_wav_{:d}_slice_{:d}_{:s}"
                            png_name = fmt.format(process_level, wave_no, slice_no, Ipc.tag)
                            png_folder = Filer.png_path + 'cube/' + Ipc.tag + '/'
                            png_folder = Filer.get_folder(png_folder)
                            png_path = png_folder + png_name
                            Plot.collage(pdp[wave_no][0:4], parameters, png_path=png_path)

                        print()
                        data_folder = date_stamp + folder_name + wave_tag
                        configuration = FitsIo.read_param_file(date_stamp, data_folder)
                        _, wave, _, _, order, im_pix_size = configuration
                        dw_lmspix = Util.find_dispersion(traces, configuration)

                        # Calculate dispersion and centre wavelength for each order.
#                        transform = Util.get_polyfit_transform(poly, ech_bounds, configuration)

                        ipc_factor, srp_lin, srp_lin_err = None, None, None
                        result = ''
                        for axis in axes:
                            axis_tag = '_' + axis[0:4]

                            ee_data = Analyse.eed(pdp[wave_no], axis, oversample=pdp_oversampling,
                                                  debug=False, log10sampling=True,
                                                  normalise='to_average')
                            lsf_data = Analyse.lsf(pdp[wave_no], axis, oversample=pdp_oversampling,
                                                   debug=False,
                                                   v_coadd=3.0 * pdp_oversampling,
                                                   u_radius=10.0 * pdp_oversampling)
                            data_id = date_stamp, slice_subfolder, Ipc.tag, process_level, wave_tag, n_mcruns_tag, axis
                            data_type = 'ee_dfp_' + axis
                            Filer.write_profiles(data_type, data_id, ee_data)
                            data_type = 'lsf_dfp_' + axis
                            Filer.write_profiles(data_type, data_id, lsf_data)

                            key_line_widths = np.zeros((3, 3))

                            obs_per = pdp[wave_no][0]
                            per_gauss, per_linear = Analyse.find_fwhm(obs_per, pdp_oversampling,
                                                                      debug=False)
                            _, fwhm_per_lin, _, xl, xr, yh = per_linear
                            key_line_widths[0, :] = [fwhm_per_lin, xl, xr]

                            obs_des = pdp[wave_no][1]
                            des_gauss, des_linear = Analyse.find_fwhm(obs_des, pdp_oversampling,
                                                                      debug=False)
                            _, fwhm_des_lin, _, xl, xr, yh = des_linear
                            key_line_widths[1, :] = [fwhm_des_lin, xl, xr]

                            # Find line widths of Monte-Carlo data
                            mc_obs_list = pdp[wave_no][2:]
                            mc_gauss, mc_linear = Analyse.find_fwhm_multi(mc_obs_list, pdp_oversampling,
                                                                          debug=False)
                            _, mc_fit, mc_fit_err = mc_gauss
                            fwhm_mc_gau, fwhm_mc_gau_err = mc_fit[1], mc_fit_err[1]
                            _, fwhm_lin_mc, fwhm_lin_mc_err, xl, xr, yh = mc_linear
                            key_line_widths[2, :] = [fwhm_lin_mc, xl, xr]

                            plot_profiles = True
                            if plot_profiles and slice_no == tgt_slice_no:
                                png_name = "lsf_{:s}_wave_{:d}".format(axis, wave_no)
                                png_folder = Filer.png_path + 'profiles/lsf/' + Ipc.tag
                                Filer.get_folder(png_folder)
                                png_path = png_folder + png_name
                                Plot.plot_lsf(lsf_data, wave, dw_lmspix, 'lsf',
                                              Ipc.tag, key_line_widths,
                                              hwlim=6.0, plot_all=True, png_path=png_path)

                            axis, xlms, ee_per, ee_des, ee_mean, ee_rms, ee_all = ee_data

                            ee_ref_radius, ee_axis_refs = Analyse.find_ee_axis_references(wave, ee_data)
                            summary_kv_pairs = []
                            for key in ee_axis_refs:
                                ee_val = ee_axis_refs[key]
                                summary_kv_pairs.append((key, ee_val))
                            result += Summariser.get_value_text(summary_kv_pairs)
                            plot_profiles = True
                            if plot_profiles and slice_no == tgt_slice_no:
                                png_name = "ee_{:s}_wave_{:d}_{:s}".format(axis, wave_no, Ipc.tag)
                                png_folder = Filer.png_path + '/profiles/lsf/'
                                Filer.get_folder(png_folder)
                                png_path = png_folder + png_name
                                Plot.plot_ee(ee_data, wave, ee_ref_radius, ee_axis_refs,
                                             'EES', Ipc.tag,
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

                        summary_kv_pairs = [('wave', wave), ('order', order),
                                            ('srp_mc_lin', srp_lin), ('srp_mc_lin_err', srp_lin_err),
                                            ('srp_mc_gau', srp_gau), ('srp_mc_gau_err', srp_gau_err)]
                        pre_result = Summariser.get_value_text(summary_kv_pairs)
                        result = pre_result + result
                        summary.append(result)

                        strips = Analyse.extract_cube_strips(pdp[wave_no], pdp_oversampling)
                        cube[wave_no, :, :, strip_col] = strips
                        fmt = "Wave={:d}/{:d}, Strip_col={:d}/{:d} -> {:s}"
                        print(fmt.format(wave_no + 1, n_wavelengths, strip_col + 1, n_cube_cols, process_level))

                    proc_cube[process_level] = cube
                    for i in range(len(summary)):
                        print(summary[i])
                    Summariser.write_summary(process_level, slice_subfolder, summary)

            for process_level in proc_cube:
                im_oversampling = Globals.get_im_oversampling(process_level)
                cube = proc_cube[process_level]
                FitsIo.write_cube(process_level, Ipc.tag, waves, im_oversampling, cube)

        png_path = Filer.get_png_path('figures', 'raw_zemax1', wave_no, slice_no)
        Plot.images(fig1_obs_list[0:4],
                    figsize=[8, 10], nrowcol=(4, 1), shrink=1.0, colourbar=False,
                    title='Fig 1a', do_log=True, png_path=png_path)
        png_path = Filer.get_png_path('figures', 'raw_zemax2', wave_no, slice_no)
        Plot.images(fig1_obs_list[4:5],
                    figsize=[8, 10], nrowcol=(4, 1), shrink=1.0, colourbar=False,
                    title='Fig 1b', do_log=True, png_path=png_path)
        png_path = Filer.get_png_path('figures', 'raw_zemax3', wave_no, slice_no)
        Plot.images(fig1_obs_list[5:9],
                    figsize=[8, 10], nrowcol=(4, 1), shrink=1.0, colourbar=False,
                    title='Fig 1c', do_log=True, png_path=png_path)
        print()
        return

    @staticmethod
    def process(config, inter_pixels):
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
                        png_folder = Filer.png_path + 'cube'
                        png_folder = Filer.get_folder(png_folder)
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
                        png_path = Filer.get_png_path('cube_eer', process_level, w, -1)
                        Plot.plot_ee(eer_data, wave, ee_ref_radius, ee_refs, title, Ipc.tag,
                                     png_path=png_path, plot_all=True)

                    data_id = dataset, '', '', 'proc_zemax', '', '', axis
                    data_type = 'ee_' + 'spatial'
                    fmt = "{:s}cube_ees_{:s}_wav{:d}_{:s}_{:s}.csv"
                    set_path = fmt.format(Filer.cube_path, process_level, w, axis, Ipc.tag)
                    Filer.write_profiles(data_type, data_id, eer_data, set_path=set_path)
            plot_cube_strehls = True
            if plot_cube_strehls:
                png_folder = Filer.png_path + '/cube_strehls'
                png_folder = Filer.get_folder(png_folder)
                png_name = "strehl_{:s}".format(process_level)
                png_path = png_folder + png_name

#                png_path = Filer.get_png_path('cube_strehls', strehl_process_level, -1, -1)
                title = 'Reconstructed {:s} image strehl ratio'.format(process_level)
                Plot.cube_strehls(waves, strehl_list, title, png_path=png_path)
        return
