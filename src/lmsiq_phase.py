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
        mc_bounds, inter_pixels = process_control
        _, _, date_stamp, _, _, _ = iq_filer.model_configuration
        tgt_slice_no, slice_radius = data_identifier['cube_slice_bounds']

        # Set up spectral shifts in detector pixels
        det_shift_start, det_shift_end, det_shift_increment = -1.0, +1.1, 0.1
        det_shifts = np.arange(det_shift_start, det_shift_end, det_shift_increment)
        n_shifts = len(det_shifts)

        # Dictionary of photometry rms values and centroid phase locations for both ipc settings
        for inter_pixel in inter_pixels:
            Ipc.set_inter_pixel(inter_pixel)
            ipc_tag = Ipc.tag
            # Initialise wavelength dependent value lists
            waves = []
            dataset_indices = []
            dataset_idx = 0

            uni_par = image_manager.unique_parameters
            config_nos = kwargs.get('config_nos', None)
            if config_nos is None:
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
                    for tgt_spifu_no in spifu_nos:
                        dataset_indices.append(dataset_idx)
                        dataset_idx += 1
                        selection = {'config_no': config_no,
                                     'field_no': field_no,
                                     'slice_no': tgt_slice_no,
                                     'spifu_no': tgt_spifu_no,
                                     'mc_bounds': mc_bounds}
                        image_list, ds_dict = image_manager.load_dataset(iq_filer,
                                                                         selection,
                                                                         # xy_shift=(10, -5, -5),
                                                                         debug=False)
                        im_pix_size = ds_dict['im_pix_size']

                        n_images = len(image_list)

                        phase_labels = ['perfect', 'design']
                        mc_start, mc_end = ds_dict['mc_bounds']
                        for mc_no in range(mc_start, mc_end+1):
                            phase_labels.append("MC_{:04d}".format(mc_no))

                        wave = ds_dict['wavelength']
                        waves.append(wave)
                        oversampling = int(Detector.det_pix_size / im_pix_size)
                        xcen_obs = {'phase shift': det_shifts}
                        xcen_obs_rms = {}
                        ycen_obs = {'phase shift': det_shifts}
                        xfwhm_obs = {'phase shift': det_shifts}
                        phot_obs = {'phase shift': det_shifts}
                        shape = n_shifts, n_images
                        xcen_shift, ycen_shift = np.zeros(shape), np.zeros(shape)
                        xfwhm_shift, phot_shift = np.zeros(shape), np.zeros(shape)
                        phot_obs_rms = {}
                        for img_idx, img_in in enumerate(image_list):
                            fmt = "\r- configuration {:d}, field {:d} of {:d}, " +\
                                  "slice {:d}, spifu {:d}, model {:03d} of {:03d}"
                            print(fmt.format(config_no,
                                  field_no, len(field_nos),
                                  tgt_slice_no, tgt_spifu_no, img_idx + 1, n_images),
                                  end="", flush=True)
                            obs_key = phase_labels[img_idx]
                            for shift_idx, det_shift in enumerate(det_shifts):
                                im_shift = det_shift * oversampling
                                im_shifted = Phase.sub_pixel_shift(img_in, 'spectral', im_shift,
                                                                   resolution=50, debug=False)
                                im_ipc = Ipc.apply(im_shifted, oversampling) if inter_pixel else im_shifted
                                im_det = Detector.measure(im_ipc, im_pix_size)

                                # Find the FWHM and <signal> in the detector plane
                                xgauss_ipc, _ = Analyse.find_fwhm(im_ipc, oversample=oversampling,
                                                                  debug=False, axis=0)
                                xgauss_det, _ = Analyse.find_fwhm(im_det, oversample=1,
                                                                  debug=False, axis=0)
                                _, xfwhm_ipc, xcen_ipc = xgauss_ipc[1]
                                _, xfwhm_det, xcen_det = xgauss_det[1]

                                ygauss, _ = Analyse.find_fwhm(im_det, oversample=oversampling,
                                                              debug=False, axis=1)
                                is_error, yfit, yfit_err = ygauss
                                if is_error:
                                    continue
                                _, _, ycen = yfit
                                method = 'full_image'          # Valid methods are 'aperture' or 'full_image'
                                ap_pos, ap_width, ap_height = (xcen_det, ycen), 16., 16.
                                phot = Analyse.find_phot(im_det,
                                                         method=method,
                                                         method_parameters=(ap_pos, ap_width, ap_height))

                                xcen_shift[shift_idx, img_idx] = xcen_det - det_shift
                                ycen_shift[shift_idx, img_idx] = ycen
                                xfwhm_shift[shift_idx, img_idx] = xfwhm_det
                                phot_shift[shift_idx, img_idx] = phot
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

                        png_folder = iq_filer.output_folder + '/phase'
                        png_folder = iq_filer.get_folder(png_folder)

                        fmt = "{:s}_{:s}_{:02d}_field{:02d}_tgt_slice{:02d}_spifu_{:02d}"
                        phase_plots = {'xcentroids': xcen_shift, 'ycentroids': ycen_shift,
                                       'photometry': phot_shift, 'xfwhm': xfwhm_shift}
                        for key in phase_plots:
                            png_name = fmt.format(key, ipc_tag, config_no, field_no, tgt_slice_no, tgt_spifu_no)
                            data_tuple = det_shifts, phase_plots[key]
                            png_path = png_folder + '' + png_name
                            title = "{:s}, ".format(date_stamp) + png_name
                            Plot.phase_shift(key, data_tuple, ds_dict, ipc_tag, title=title, png_path=png_path)
            print()
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
