import numpy as np
from lms_detector import Detector
from lms_ipc import Ipc
from lmsiq_plot import Plot
from lmsiq_analyse import Analyse
from lms_util import Util


class Phase:

    def __init__(self):
        return

    @staticmethod
    def process(data_identifier, process_control, iq_filer, image_manager, **kwargs):
        """ Read in a Zemax image data set and generate images sampled at the detector with
        or without intra-pixel diffusion applied.
        """
        plot_level = kwargs.get('plot_level', 1)        # 0=none, 1=some, 2=all
        mc_bounds, inter_pixels = process_control

        tgt_slice_no, slice_radius = data_identifier['cube_slice_bounds']
        tgt_spifu_no = 1

        # Set up spectral shifts in detector pixels
        det_shift_start, det_shift_end, det_shift_increment = -1.0, +1.1, 0.1
        det_shifts = np.arange(det_shift_start, det_shift_end, det_shift_increment)
        n_shifts = len(det_shifts)

        # Dictionary of photometry rms values and centroid phase locations for both ipc settings
        ipc_labels = []

        phot_rms_ipc, xcen_rms_ipc = {}, {}
        for inter_pixel in inter_pixels:
            Ipc.set_inter_pixel(inter_pixel)
            ipc_labels.append(Ipc.tag)
            # Initialise wavelength dependent value lists
            waves = []
            phot_rms_dataset, xcen_rms_dataset = [], []
            dataset_indices = []
            dataset_idx = 0

            uni_par = image_manager.unique_parameters
            config_nos = uni_par['config_nos']
            field_nos = uni_par['field_nos']
            slice_nos = uni_par['slice_nos']
            spifu_nos = uni_par['spifu_nos']

            print()
            print("Detector diffusion = {:s}".format(str(inter_pixel)))
            Util.print_list('configurations', config_nos)
            Util.print_list('field numbers ', field_nos)
            Util.print_list('slice numbers ', slice_nos)
            Util.print_list('spifu numbers ', spifu_nos)

            for config_no in config_nos:
                for field_no in field_nos:
                    dataset_indices.append(dataset_idx)
                    dataset_idx += 1

                    obs_list, obs_dict = image_manager.load_dataset(iq_filer,
                                                                    config_no=config_no,
                                                                    field_no=field_no,
                                                                    slice_no=tgt_slice_no,
                                                                    spifu_no=tgt_spifu_no,
                                                                    mc_bounds=mc_bounds,
                                                                    debug=False
                                                                    )
                    n_obs = len(obs_list)

                    phase_labels = ['perfect', 'design']
                    mc_start, mc_end = obs_dict['mc_bounds']
                    for mc_no in range(mc_start, mc_end+1):
                        phase_labels.append("MC_{:04d}".format(mc_no))

                    im_pix_size = obs_dict['im_pix_size']

                    wave = obs_dict['wavelength']
                    waves.append(wave)
                    oversampling = int(Detector.det_pix_size / im_pix_size)
                    det_oversampling = 1
                    xcen_obs = {'phase shift': det_shifts}
                    xcen_obs_rms = {}
                    ycen_obs = {'phase shift': det_shifts}
                    xfwhm_obs = {'phase shift': det_shifts}
                    phot_obs = {'phase shift': det_shifts}
                    phot_obs_rms = {}
                    for obs_no, obs_1 in enumerate(obs_list):
                        fmt = "\r- configuration {:d}, field {:d} of {:d}, " +\
                              "slice {:d}, model {:03d} of {:03d}"
                        print(fmt.format(config_no,
                              field_no, len(field_nos), tgt_slice_no, obs_no + 1, n_obs),
                              end="", flush=True)
                        obs_key = phase_labels[obs_no]
                        xcen_shift, ycen_shift = np.zeros(n_shifts), np.zeros(n_shifts)
                        xfwhm_shift, phot_shift = np.zeros(n_shifts), np.zeros(n_shifts)
                        for shift_no, det_shift in enumerate(det_shifts):
                            im_shift = det_shift * oversampling
                            obs_2 = Phase.sub_pixel_shift(obs_1, 'spectral', im_shift,
                                                          resolution=50, debug=False)
                            obs_3 = Ipc.convolve(obs_2, oversampling) if inter_pixel else obs_2
                            if plot_level > 1:
                                sf_fmt = "{:s}/slice{:02d}/wav{:02d}/model{:03d}/"
                                png_sub_folder = sf_fmt.format(Ipc.tag, tgt_slice_no, config_no, obs_no)
                                png_folder = iq_filer.iq_png_folder + 'zemax/' + png_sub_folder
                                png_folder = iq_filer.get_folder(png_folder)
                                png_name = png_sub_folder.replace('/', '_') + "obs{:03d}".format(shift_no)
                                title = "{:s}, ".format(date_stamp) + png_name
                                png_path = png_folder + png_name
                                Plot.images([obs_3], nrowcol=(1, 1), title=title,
                                            shrink=0.25, png_path=png_path)

                            obs_4 = Detector.measure(obs_3, im_pix_size)

                            # Find the FWHM and <signal> in the detector plane (obs_4)
                            xgauss, _ = Analyse.find_fwhm(obs_4, oversample=det_oversampling,
                                                          debug=False, axis=0)
                            is_error, xfit, xfit_err = xgauss
                            if is_error:
                                continue
                            _, xfwhm, xcen = xfit
                            ygauss, _ = Analyse.find_fwhm(obs_4, oversample=det_oversampling,
                                                          debug=False, axis=1)
                            is_error, yfit, yfit_err = ygauss
                            if is_error:
                                continue
                            _, _, ycen = yfit
                            method = 'full_image'          # Valid methods are 'aperture' or 'full_image'
                            ap_pos, ap_width, ap_height = (xcen, ycen), 16., 16.
                            phot = Analyse.find_phot(obs_4,
                                                     method=method,
                                                     method_parameters=(ap_pos, ap_width, ap_height))

                            xcen_shift[shift_no] = xcen - det_shift
                            ycen_shift[shift_no] = ycen
                            xfwhm_shift[shift_no] = xfwhm
                            phot_shift[shift_no] = phot
                            det_shift += det_shift_increment

                        xcen_shift_mean = np.mean(xcen_shift)
                        xcen_shift_rms = np.std(xcen_shift)
                        ycen_shift_mean = np.mean(ycen_shift)
                        phot_shift_mean = np.mean(phot_shift)
                        phot_shift_rms = np.std(phot_shift)
                        phot_obs[obs_key] = phot_shift / phot_shift_mean
                        xcen_obs_rms[obs_key] = xcen_shift_rms
                        phot_obs_rms[obs_key] = phot_shift_rms
                        xcen_obs[obs_key] = xcen_shift / xcen_shift_mean
                        ycen_obs[obs_key] = ycen_shift / ycen_shift_mean
                        xfwhm_obs[obs_key] = xfwhm_shift

                    ipc_tag = Ipc.tag
                    phot_path = iq_filer.write_phase_data(phot_obs, 'photometry', config_no, ipc_tag)
                    xcen_path = iq_filer.write_phase_data(xcen_obs, 'xcentroids', config_no, ipc_tag)
                    ycen_path = iq_filer.write_phase_data(ycen_obs, 'ycentroids', config_no, ipc_tag)
                    xfwhm_path = iq_filer.write_phase_data(xfwhm_obs, 'xfwhm_gau', config_no, ipc_tag)

                    Plot.phase_shift('photometry', phot_obs, obs_dict, wave, ipc_tag, png_path=phot_path)
                    Plot.phase_shift('xcentroid', xcen_obs, obs_dict, wave, ipc_tag, png_path=xcen_path)
                    Plot.phase_shift('ycentroid', ycen_obs, obs_dict, wave, ipc_tag, png_path=ycen_path)
                    Plot.phase_shift('xfwhm', xfwhm_obs, obs_dict, wave, ipc_tag, png_path=xfwhm_path)

                    phot_rms_dataset.append(phot_obs_rms)
                    xcen_rms_dataset.append(xcen_obs_rms)

            print()
            phot_rms_ipc[ipc_tag] = phot_rms_dataset
            xcen_rms_ipc[ipc_tag] = xcen_rms_dataset

        data_type = 'photometry_rms'
        png_name = data_type + "_config{:02d}_{:s}".format(config_no, ipc_tag)
        png_path = iq_filer.get_folder(iq_filer.output_folder + 'phase/' + data_type) + png_name
        date_stamp = data_identifier['iq_date_stamp']
        title = date_stamp + ', photometric rms variation'
        Plot.rms_vals(dataset_indices, phot_rms_ipc,
                      title=title, sigma=1., png_path=png_path)

        data_type = 'centroid_rms'
        png_name = data_type + "_config{:02d}_{:s}".format(config_no, ipc_tag)
        png_path = iq_filer.get_folder(iq_filer.output_folder + 'phase/' + data_type) + png_name
        title = date_stamp + ', LSF centroid rms variation'
        Plot.rms_vals(dataset_indices, xcen_rms_ipc,
                      title=title, sigma=1., png_path=png_path)
        print()
        return

    @staticmethod
    def sub_pixel_shift(image, axis, sp_shift, **kwargs):
        debug = kwargs.get('debug', False)
        resolution = kwargs.get('resolution', 10)       # Number of sub-pixels per image pixel

        image_in = np.array(image)                      # Clone input to avoid corrupting observation
        if axis == 'spectral':                          # Rotate input image
            np.moveaxis(image_in, 0, 1)
        nrows, ncols = image_in.shape
        ncols_ss = resolution * ncols
        image_ss = np.zeros((nrows, ncols_ss))          # Create super-sampled image
        col_shift = round(sp_shift * resolution)
        if debug:
            fmt = "Shifting {:s} axis image, ({:d}x) oversampled by {:d} columns"
            print(fmt.format(axis, resolution, col_shift))
        for col in range(0, ncols):     # Map shifted image data into super-sampled image
            c1 = col * resolution + col_shift
            c2 = c1 + resolution
            c1 = c1 if c1 > 0 else 0
            c2 = c2 if c2 < ncols_ss else ncols_ss - 1
            for c_ss in range(c1, c2):
                image_ss[:, c_ss] = image_in[:, col]
        image_out = np.zeros(image_in.shape)
        for col in range(0, ncols):     # Resample super-sampled image onto output image
            col_ss = col * resolution
            strip = np.mean(image_ss[:, col_ss:col_ss + resolution], axis=1)
            image_out[:, col] = strip
        if axis == 'spectral':                      # Rotate input image back to original orentation
            np.moveaxis(image_out, 0, 1)
        return image_out        # , params
