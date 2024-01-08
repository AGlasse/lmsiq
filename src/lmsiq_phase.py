import numpy as np
from lms_filer import Filer
from lms_dist_util import Util
from lmsiq_fitsio import FitsIo
from lms_globals import Globals
from lms_detector import Detector
from lms_ipc import Ipc
from lmsiq_plot import Plot
from lmsiq_analyse import Analyse


class Phase:

    def __init__(self):
        return

    @staticmethod
    def process(config, inter_pixels):

        # Multi-slice, multi-wavelength, multi-Monte Carlo run configurations
        dataset, n_wavelengths, n_mcruns, zim_locs, folder_name, config_label = config
        n_mcruns_tag = "{:04d}".format(n_mcruns)
        tgt_slice_no, n_slices, slice_idents = Util.parse_slice_locations(zim_locs)

        for slice_ident in slice_idents:
            slice_no, slice_subfolder, slice_label = slice_ident
            if slice_no != tgt_slice_no:
                continue

            # Set up spectral shifts in detector pixels
            det_shift_start, det_shift_end, det_shift_increment = -1.0, +1.1, 0.1
            det_shifts = np.arange(det_shift_start, det_shift_end, det_shift_increment)
            n_shifts = len(det_shifts)

            # Dictionary of photometry rms values and centroid phase locations for both ipc settings
            phot_rms, cen_rms = {}, {}
            ipc_labels = []
            for inter_pixel in inter_pixels:
                Ipc.set_inter_pixel(inter_pixel)
                ipc_labels.append(Ipc.tag)
                phot_rms_vals, cen_rms_vals = [], []
                waves = []
                for wave_no in range(0, n_wavelengths):
                    wave_tag = "{:02d}/".format(wave_no)

                    n_obs = n_mcruns + 2  # Include the 'perfect' and 'design' cases at the start of the obs list
                    block_shape = n_shifts, n_obs + 1       # Sub-pixel shift is written to col 0
                    xcen_block, ycen_block = np.zeros(block_shape), np.zeros(block_shape)
                    fwhm_block, phot_block = np.zeros(block_shape), np.zeros(block_shape)
                    xcen_block[:, 0], ycen_block[:, 0] = det_shifts, det_shifts
                    fwhm_block[:, 0], phot_block[:, 0] = det_shifts, det_shifts

                    zim_data_folder = dataset + folder_name + wave_tag
                    # For first dataset, read in parameters from text file and generate IPC and IPG kernels.
                    zemax_configuration = FitsIo.read_param_file(dataset, zim_data_folder)
                    _, wave, _, _, order, im_pix_size = zemax_configuration
                    waves.append(wave)
                    slice_folder = zim_data_folder + slice_subfolder
                    # Select files to analyse
                    obs_list = FitsIo.load_dataset(dataset, slice_folder, n_mcruns)
                    oversampling = Globals.get_im_oversampling('raw_zemax')
                    det_oversampling = Globals.get_im_oversampling('proc_detector')
                    for res_col, obs_1 in enumerate(obs_list):
                        fmt = "\r- Diffusion= {:s}, slice {:d}, wavelength {:02d} of {:02d}, model {:03d} of {:03d}"
                        print(fmt.format(str(inter_pixel), slice_no, wave_no + 1,
                              n_wavelengths, res_col + 1, n_mcruns + 2),
                              end="", flush=True)
                        xcen_2_list, xcen_4_list = None, None
                        for res_row, det_shift in enumerate(det_shifts):
                            im_shift = det_shift * oversampling
                            obs_2 = Phase.sub_pixel_shift(obs_1, 'spectral', im_shift,
                                                          resolution=50, debug=False)
                            obs_3 = Ipc.convolve(obs_2) if inter_pixel else obs_2
                            # fmt = "../output/zem_{:s}_{:d}"
                            # Plot.images([obs_3], nrowcol=(1, 1), title='Ipc Test',
                            #             shrink=0.25, png_path=fmt.format(Ipc.tag, res_row))

                            obs_4 = Detector.measure(obs_3)
                            # fmt = "../output/det_{:s}_{:d}"
                            # Plot.images([obs_4], nrowcol=(1, 1), title='Ipc Test',
                            #             shrink=0.25, png_path=fmt.format(Ipc.tag, res_row))

                            # Find the FWHM and <signal> in the detector plane (obs_4)
                            xgauss, _ = Analyse.find_fwhm(obs_4, det_oversampling,
                                                          debug=False, axis=0)
                            is_error, xfit, xfit_err = xgauss
                            if is_error:
                                continue
                            _, xfwhm, xcen = xfit
                            if xcen_4_list is None:
                                xcen_4_list = []
                            xcen_4_list.append(xcen)

                            ygauss, _ = Analyse.find_fwhm(obs_4, det_oversampling,
                                                          debug=False, axis=1)
                            is_error, yfit, yfit_err = ygauss
                            if is_error:
                                continue
                            _, _, ycen = yfit

                            method = 'full_image'          # 'aperture' 'full_image'
                            ap_pos, ap_width, ap_height = (xcen, ycen), 16., 16.
                            phot = Analyse.find_phot(obs_4,
                                                     method=method,
                                                     method_parameters=(ap_pos, ap_width, ap_height))

                            # fmt = "{:10d}, {:10.5e}"
                            # print(fmt.format(res_row, phot))     # np.sum(obs_4[0])))

                            xcen_block[res_row, res_col + 1] = xcen
                            ycen_block[res_row, res_col + 1] = ycen
                            fwhm_block[res_row, res_col + 1] = xfwhm
                            phot_block[res_row, res_col + 1] = phot
                            det_shift += det_shift_increment
                    process_level = 'proc_detector'
                    data_id = dataset, slice_subfolder, Ipc.tag, process_level, wave_tag, n_mcruns_tag, 'N/A'
                    Filer.write_phase_data('xcentroids', data_id, det_shifts, xcen_block)
                    Filer.write_phase_data('ycentroids', data_id, det_shifts, ycen_block)
                    Filer.write_phase_data('xfwhm_gau', data_id, det_shifts, fwhm_block)
                    Filer.write_phase_data('photometry', data_id, det_shifts, phot_block)

                    centroids = np.array(xcen_block)
                    _, n_cols = centroids.shape
                    for col in range(1, n_cols):
                        centroids[:, col] = centroids[:, col] - det_shifts
                    cen_rms_val = np.std(centroids[:, 3:])      # Only use MC data in rms calculation
#                    cen_pk_pk = np.amax(centroids[:, 3:]) - np.amin(centroids[:, 3:])
                    cen_rms_vals.append(cen_rms_val)

                    phot_mean = np.mean(phot_block[:, 3:])
                    phot_norms = phot_block[:, 3:] / phot_mean
                    phot_rms_val = np.std(phot_norms)
                    phot_rms_vals.append(phot_rms_val)

                    png_folder = Filer.png_path + 'phase/photom'
                    png_folder = Filer.get_folder(png_folder)
                    png_name = "photom_change_{:02d}_{:s}".format(wave_no, Ipc.tag)
                    png_path = png_folder + png_name
                    Plot.phase_shift('photometry', phot_block, wave,
                                     mc_only=True, png_path=png_path)
                    png_folder = Filer.png_path + 'phase/centroid'
                    png_folder = Filer.get_folder(png_folder)
                    png_name = "centroid_shift_{:02d}_{:s}".format(wave_no, Ipc.tag)
                    png_path = png_folder + png_name
                    Plot.phase_shift('centroids', centroids, wave,
                                     mc_only=True, png_path=png_path)

                phot_rms[Ipc.tag] = phot_rms_vals
                cen_rms[Ipc.tag] = cen_rms_vals
            png_folder = Filer.png_path + '/phase'
            png_folder = Filer.get_folder(png_folder)
            png_name = 'photometry_rms'
            png_path = png_folder + png_name
            Plot.rms_vals(waves, phot_rms,
                          title='photometric variation', sigma=1., png_path=png_path)
            png_folder = Filer.png_path + '/phase'
            png_folder = Filer.get_folder(png_folder)
            png_name = 'centroid_rms'
            png_path = png_folder + png_name
            Plot.rms_vals(waves, cen_rms,
                          title='centroid variation', sigma=1., png_path=png_path)
        print()
        return

    @staticmethod
    def sub_pixel_shift(observation, axis, sp_shift, **kwargs):
        debug = kwargs.get('debug', False)
        resolution = kwargs.get('resolution', 10)       # Number of sub-pixels per image pixel

        image, params = observation
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
        return image_out, params
