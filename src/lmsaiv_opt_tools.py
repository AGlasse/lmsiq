import math
import numpy as np
import astropy.units as u
import copy
import scipy
from scipy.optimize import curve_fit, OptimizeWarning
from lms_globals import Globals


class OptTools:


    def __init__(self):
        return

    @staticmethod
    def copy_mosaic(mosaic, clear_data=False, copy_name=''):
        file_name, hdr, hdus = mosaic
        moscopy_hdus, moscopy_hdr = [], None
        for hdu in hdus:
            moscopy_hdr = copy.deepcopy(hdu.header)
            moscopy_hdu = hdu.copy()
            if clear_data is not None:
                moscopy_hdu.data *= 0.
            moscopy_hdus.append(moscopy_hdu)
        moscopy_name = file_name if copy_name == '' else copy_name
        moscopy = moscopy_name, moscopy_hdr, moscopy_hdus
        return moscopy

    @staticmethod
    def dark_stats(mosaics):
        for mosaic in mosaics:
            file_name, hdr, hdus = mosaic
            dit = hdr['HIERARCH ESO DET DIT']
            ndit = hdr['HIERARCH ESO DET NDIT']
            t_int = dit * ndit

            print()
            print("File = {:s}".format(file_name))
            print("Signal distribution statistics, integration time = {:10.1f}".format(t_int))
            fmt = "{:>8s},{:>10s},{:>10s},{:>10s},{:>10s}"
            print(fmt.format('Detector', 'median', 'stdev', 'median', 'Rd_Noise'))
            print(fmt.format('No.', 'ADU', 'ADU', 'el/sec.', 'el.'))
            fmt = "{:8d},{:10.3f},{:10.3f},{:10.3f},{:10.3f}"

            for i, hdu in enumerate(hdus):
                el_adu = hdu.header['HIERARCH ESO DET3 CHIP GAIN']
                median = np.median(hdu.data)
                stdev = np.std(hdu.data)
                median_current = median * el_adu / dit
                rd_noise = stdev * el_adu / math.sqrt(2. / ndit)
                text = fmt.format(i + 1, median, stdev, median_current, rd_noise)
                print(text)
        return

    @staticmethod
    def flood_stats(mosaic):
        """ Calculate fov and return dictionary of slice_bounds and profiles used to calculate them.
        """
        file_name, hdr, hdus = mosaic
        opticon = hdr['HIERARCH ESO INS MODE']

        # Set up slice map object to hold slice images
        slice_map = OptTools.copy_mosaic(mosaic, clear_data=True, copy_name='slice_map')

        u.arcsec2 = u.arcsec * u.arcsec

        dark_pctile = 10.
        bright_pctile = 90.         # Choose the bright pixel limit to avoid hot pixels.
        alpha_cut = 0.5
        print()
        print("File = {:s}".format(file_name))
        fmt = "Dark pixels defined as those < {:.0f}th percentile signal level"
        print(fmt.format(dark_pctile))
        fmt = "Illuminated pixels defined as those brighter than the {:.0f}th percentile signal level"
        print(fmt.format(bright_pctile))
        fmt = "alpha extent of each slice defined as distance between {:3.2f} of bright level"
        print(fmt.format(alpha_cut))
        fmt = "Illuminated pixel x slice fov = {} x {} mas"
        print(fmt.format(Globals.alpha_pix, Globals.beta_slice))
        profile_cols = {1: [600, 800, 1000, 1200], 3: [600, 700, 800, 1200],
                        2: [1000, 1800, 1900, 2000], 4: [1000, 1800, 1900, 2000]}

        print()
        fmt = "{:>10s},{:>10s},{:>10s},{:>10s}"
        print(fmt.format('Detector', 'dark',  'bright',    'illum.'))
        print(fmt.format('        ', 'level', 'level',        'fov'))
        print(fmt.format('        ', 'DN',       'DN',  '[sq_asec]'))
        fmt = "{:>10d},{:>10.2e},{:>10.2e},{:>10.3f}"

        # Separate slices by finding cuts in d_signal / d_row
        n_slices = 28 if '_nom_' in file_name else 3
        n_spifus = 0 if '_nom_' in file_name else 6

        profiles = []
        alpha_det = [0.]*4
        for hdu in hdus:
            det_no = int(hdu.header['ID'])
            det_idx = det_no - 1
            slice_coords = {'det_no': det_no, 'slice_nos': [], 'cols': [], 'row_mins': [], 'row_maxs': []}

            n_profiles = len(profile_cols[det_no])
            dark_level, bright_level = 0., 0.           # Average signal cut levels for this detector
            for pr_col in profile_cols[det_no]:
                pr_col_hw = 2                           # Co-add 2 x pr_col_hw + 1 centred on pr_col.
                col1, col2 = pr_col - pr_col_hw, pr_col + pr_col_hw

                slice_no = 1 if det_no in [3, 4] else 15  # Starting slice number (bottom row)
                spifu_no = 0
                if n_spifus > 0:
                    slice_no = 12
                    spifu_no = 1 if det_no in [3, 4] else 4
                y_vals = np.nanmean(hdu.data[:, col1:col2+1], axis=1)
                dark_level = np.nanpercentile(y_vals, dark_pctile)
                bright_level = np.nanpercentile(y_vals, bright_pctile)
                on_rows, off_rows = [], []
                row_off = 0
                cut_level = bright_level * alpha_cut
                slice_count = 0
                while True:
                    # Find next bright pixel
                    on_indices = np.argwhere(y_vals[row_off:] > cut_level)
                    if len(on_indices) < 1:       # No more bright pixels found.
                        break
                    row_on = row_off + on_indices[0][0]
                    r_on = np.interp(cut_level, [y_vals[row_on-1], y_vals[row_on]], [row_on-1, row_on])
                    off_indices = np.argwhere(y_vals[row_on:] < cut_level)
                    row_off = row_on + off_indices[0][0]
                    slice_coords['cols'].append(pr_col)
                    slice_coords['row_mins'].append(row_on)
                    slice_coords['row_maxs'].append(row_off)
                    slice_coords['slice_nos'].append(slice_no)
                    r_off = np.interp(cut_level, [y_vals[row_off], y_vals[row_off-1]], [row_off, row_off-1])

                    alpha_slice = (row_off - r_on) * Globals.alpha_pix
                    alpha_det[det_idx] += alpha_slice
                    on_rows.append(r_on)
                    off_rows.append(r_off)

                    if n_spifus == 0:
                        slice_no += 1
                    else:                       # MSA set to 'extended'
                        slice_count += 1
                        slice_no += 1
                        if slice_count % n_slices == 0:
                            slice_no -= n_slices
                            if spifu_no != 0:
                                spifu_no += 1
                x_indices = np.arange(0, len(y_vals))
                title = "det {:d}, cols {:d}-{:d}".format(det_no, col1, col2)
                on_points = on_rows, [cut_level] * len(on_rows), 'green'
                off_points = off_rows, [cut_level] * len(off_rows), 'blue'
                profile = title, x_indices, y_vals, [on_points, off_points]
                profiles.append(profile)
                # Calculate total length of illuminated slices.
                illum_rows_sum = 0.
                for row_on, row_off in zip(on_rows, off_rows):
                    illum_rows = row_off - row_on
                    illum_rows_sum += illum_rows

            # Create the slice map from the table of slice bounds.
            poly_degree = 2 if n_profiles > 3 else n_profiles - 1

            slice_nos = np.array(slice_coords['slice_nos'], dtype='int64')
            unique_slice_nos = np.unique(slice_nos)
            slice_map_name, slice_map_hdr, slice_map_hdus = slice_map
            slice_map_hdu = slice_map_hdus[det_idx]
            for slice_no in unique_slice_nos:
                idx = np.where(slice_no == slice_nos)[0]
                cols = np.array(slice_coords['cols'])[idx]
                row_mins = np.array(slice_coords['row_mins'])[idx]
                row_maxs = np.array(slice_coords['row_maxs'])[idx]
                row_min_fit = np.polyfit(cols, row_mins, poly_degree)
                row_max_fit = np.polyfit(cols, row_maxs, poly_degree)
                nr, nc = slice_map_hdus[det_idx].data.shape
                cs = np.arange(0, nc, 1)
                r1s = np.rint(np.polyval(row_min_fit, cs))
                r2s = np.rint(np.polyval(row_max_fit, cs))
                for c, r1, r2 in zip(cs, r1s, r2s):
                    slice_map_hdu.data[int(r1):int(r2), int(c)] = slice_no
            alpha_det[det_idx] /= n_profiles
            det_fov = alpha_det[det_idx] * Globals.beta_slice
            fmt = "{:>10d},{:>10.1f},{:>10.1f},{:>10.3f}"
            print(fmt.format(det_no, dark_level, bright_level, det_fov.to(u.arcsec2)))

        alpha_02 = alpha_det[0] + alpha_det[2]
        alpha_13 = alpha_det[1] + alpha_det[3]
        alpha_ave = 0.5 * (alpha_02 + alpha_13)
        fov = alpha_ave * Globals.beta_slice
        print()
        print("Total field of view = {:5.3f} (cf METIS-3667, shall be > 0.500 arcsec2)".format(fov.to(u.arcsec2)))
        alpha_ext, beta_ext = alpha_ave / n_slices, Globals.beta_slice * n_slices
        aspect_ratio = alpha_ext / beta_ext
        print("Aspect ratio (alpha/beta) = {:5.3f}:1 (cf METIS-3667, shall be 1:1 < ar < 2:1)".format(aspect_ratio))
        return slice_map, profiles

    # @staticmethod
    # def extract_alpha_traces(mosaic, col_start=400, col_end=1600, col_spacing=100):
    #     """ For a spectral image of a compact source, extract a set of along-column profiles
    #     col_spacing: column gap between profiles.
    #     """
    #     mos_name, mos_primary_header, mos_hdus = mosaic
    #     trace_id = 0
    #     alpha_traces = {'name': mos_name, 'det_no': [], 'trace_id': [],
    #                     'popt': [], 'popt_err': [], 'rot_angle': []}
    #     col_hw = int(col_spacing / 2)
    #     cols = list(range(col_start, col_end, col_spacing))
    #
    #     for i, hdu in enumerate(mos_hdus):
    #         # Identify the alpha traces in this detector by collapsing along rows.
    #         image = hdu.data
    #         s_row_aves = np.mean(image, axis=1)
    #         s_row_stds = np.std(image, axis=1)
    #         bgd_noise = np.median(s_row_stds)                # Background noise
    #         r_max_list = []
    #         more_traces = True
    #         text = ''
    #         while more_traces:
    #             r_max = np.argmax(s_row_aves)           # Brightest row
    #             r1, r2 = r_max - 20, r_max + 20     # Rows to use for trace analysis
    #             # Find the trace coordinates by fitting gaussians
    #             snr = s_row_aves[r_max] / bgd_noise
    #             more_traces = snr > 10
    #             if more_traces:
    #                 text += ", {:d} ".format(r_max)
    #                 r_max_list.append(r_max)
    #                 s_row_aves[r1:r2] = s_row_aves[r1]
    #         print("Detector {:d}.  Found {:d} alpha traces, at rows{:s}".format(i+1, len(r_max_list), text))
    #         r_hw = 15       # Half width in rows of strip to search for trace
    #         for r_max in r_max_list:
    #             r1, r2 = r_max - r_hw, r_max + r_hw     # Rows to use for trace analysis
    #             # Find the trace coordinates by fitting gaussians
    #             pt_row_coords, pt_col_coords = [], []
    #             for col in cols:
    #                 c1, c2 = col - col_hw, col + col_hw
    #                 z_vals = np.mean(image[r1:r2, c1:c2], axis=1)
    #                 r_vals = np.array(list(range(r1, r2)))
    #                 r_max = np.argmax(z_vals)
    #                 z_max = z_vals[r_max]
    #                 r_sigma = 1.0
    #                 p0_guess = [z_max, r1 + r_hw, r_sigma]
    #                 try:
    #                     gopt, gcov = curve_fit(Globals.gauss, r_vals, z_vals,
    #                                            p0=p0_guess)
    #                     r_gauss_peak = gopt[1]
    #                     pt_row_coords.append(r_gauss_peak)
    #                     pt_col_coords.append(col)
    #                 except:
    #                     print('Gaussian fit failed')
    #
    #             # Remove the peak to find next brightest.  Quit when snr < 100
    #             mean_row = np.mean(pt_row_coords)
    #             p0_guess = [mean_row, 0., 0., 0.]
    #             popt, pcov = curve_fit(Globals.cubic, pt_col_coords, pt_row_coords,
    #                                    p0=p0_guess)
    #             trace_id += 1
    #             alpha_traces['trace_id'].append(trace_id)
    #             alpha_traces['det_no'].append(i + 1)
    #             alpha_traces['popt'].append(popt)
    #             alpha_traces['popt_err'].append(np.sqrt(np.diag(pcov)))
    #             rot_angle = math.atan(popt[1]) * 180 / np.pi
    #             alpha_traces['rot_angle'].append(rot_angle)
    #
    #             s_row_aves[r1:r2] = s_row_aves[r1-1]
    #
    #     return alpha_traces

    @staticmethod
    def transform_detector_image(mosaic, det_no, xy_pix=(0, 0), angle=0.0):
        mos_name, mos_primary_header, mos_hdus = mosaic
        det_idx = det_no - 1
        cosa = math.cos(math.radians(angle))
        sina = math.sin(math.radians(angle))
        img = mos_hdus[det_idx].data
        tr_mat = np.array([[cosa, sina, 0.], [-sina, cosa, 0.], [0., 0., 1.]])
        rotimg = scipy.ndimage.affine_transform(img, tr_mat, cval=0.1)
        mos_hdus[det_idx].data = rotimg
        return mosaic

    @staticmethod
    def extract_det_traces(mosaic, type, slice_map, **kwargs):
        """ Extract iso-alpha or iso-lambda traces from a spectral image.
        :param mosaic: Data tuple (name, , image list
        :param type: 'alpha' or 'lambda'
        :param slice_map:
        :param kwargs:
        :return:
        """
        popt, pcov = None, None

        is_alpha = type == 'alpha'
        mos_name, mos_primary_header, mos_hdus = mosaic
        trace_idx = 0
        det_traces = {'name': mos_name, 'type': type, 'det_no': [], 'mos_idx': [],
                      'trace_idx': [], 'slice_no': [],
                      'pt_u_coords': [], 'pt_v_coords':[], 'u_mean':[], 'v_max':[],
                      'popt': [], 'popt_err': [], 'rot_angle': []}
        _, _, slice_map_hdus = slice_map
        for i, hdu in enumerate(mos_hdus):
            det_no = int(hdu.header['ID'].strip())
            mos_idx = Globals.mos_idx[det_no]
            # print(i, det_no, hdu.header['ID'])
            image = np.array(hdu.data)
            slice_map_data = np.array(slice_map_hdus[i].data)
            snu = np.unique(slice_map_data)
            slice_nos = [int(s) for s in snu if s > 0.]
            # Start by extracting the data for each slice.
            for slice_no in slice_nos:
                idx = np.argwhere(slice_no == slice_map_data)
                rs_min, rs_max = np.amin(idx[:, 0]), np.amax(idx[:, 0])
                u_off = 0 if is_alpha else rs_min
                slice_image = np.array(image[rs_min:rs_max, :])             # Extract slice image
                # We locate traces by collapsing along rows (for iso-alpha) or columns (for iso-lambda)
                # Notation, u = along trace, v = orthogonal to trace
                axis = 1 if is_alpha else 0
                v_max_list = []
                # Loop to find all traces in this slice image.
                noise = np.median(np.std(slice_image, axis=axis))         # Noise measure for slice image
                bgd = np.median(slice_image)
                repeat = True
                s_aves = np.mean(slice_image, axis=axis)
                while repeat:
                    v_max = np.argmax(s_aves)                           # Brightest row (alpha) / column (lambda)
                    v1_sig, v2_sig = v_max - 5, v_max + 5               # Rows/cols to use for trace analysis
                    v1_clr, v2_clr = v_max - 10, v_max + 10
                    signal = np.mean(s_aves[v1_sig:v2_sig])
                    snr = (signal - bgd) / noise
                    repeat = snr > 5
                    if repeat:
                        # print(det_no, slice_no, v_max, signal, bgd, noise, snr)
                        # text += ", {:d} ".format(v_max + rs_min)
                        v_max_list.append(v_max)
                        s_aves[v1_clr:v2_clr] = bgd
                if len(v_max_list) < 1:
                    if Globals.is_debug('medium'):
                        print('No traces found for det_no ', det_no, ', slice_no ', slice_no)
                    continue
                u_count = slice_image.shape[1] if is_alpha else slice_image.shape[0]
                n_samples = 10 if is_alpha else 5
                u_interval = u_count // (n_samples + 1)
                u_start, u_end = u_interval, u_count - u_interval
                u_list = list(range(u_start, u_end, u_interval))
                u_hw, v_hw = 5, 10       # Sample half width in along and across trace dimensions to fit gaussian.
                for v_max in v_max_list:
                    v1, v2 = v_max - v_hw, v_max + v_hw  # Across trace rows/cols to use for trace analysis
                    pt_u_coords, pt_v_coords = [], []
                    for u in u_list:
                        # Find the trace coordinates by fitting gaussians
                        u1, u2 = u - u_hw, u + u_hw
                        sample_image = slice_image[v1:v2, u1:u2] if is_alpha else slice_image[u1:u2, v1:v2]
                        z_vals = np.mean(sample_image, axis=axis)
                        v_vals = np.array(list(range(v1, v2)))
                        idx_max = np.argmax(z_vals)
                        z_max = z_vals[idx_max]
                        v_sigma = 1.0
                        p0_guess = [z_max, v1 + v_hw, v_sigma]
                        try:
                            pt_u_coords.append(u + u_off)
                            gopt, gcov = curve_fit(Globals.gauss, v_vals, z_vals, p0=p0_guess)
                            v_off = rs_min if is_alpha else 0
                            v_gauss_peak = gopt[1] + v_off     # Get the row/col coordinate in the image frame.
                            pt_v_coords.append(v_gauss_peak)
                        except:
                            print('Gaussian fit failed')

                    u_mean = np.mean(pt_u_coords)
                    p0_guess = [u_mean, 0., 0., 0.]
                    try:
                        popt, pcov = curve_fit(Globals.polynomial, pt_u_coords, pt_v_coords, p0=p0_guess)
                    except (RuntimeError, OptimizeWarning):
                        print('!! Error finding polynomial trace fit !!')
                    det_traces['trace_idx'].append(trace_idx)
                    det_traces['det_no'].append(det_no)
                    det_traces['mos_idx'].append(mos_idx)
                    det_traces['slice_no'].append(slice_no)
                    det_traces['popt'].append(popt)
                    det_traces['popt_err'].append(np.sqrt(np.diag(pcov)))
                    det_traces['pt_u_coords'].append(pt_u_coords)
                    det_traces['pt_v_coords'].append(pt_v_coords)
                    det_traces['u_mean'].append(u_mean)
                    det_traces['v_max'].append(v_max)
                    # Calculate rotation angle using fit at slice endpoints
                    v_start = Globals.polynomial(u_start + u_off, *popt)
                    v_end = Globals.polynomial(u_end + u_off, *popt)

                    dv_du = (v_end - v_start) / (u_end - u_start)
                    # print(u_start, u_end, v_start, v_end, dv_du)
                    rot_angle = dv_du * 180 / np.pi
                    det_traces['rot_angle'].append(rot_angle)
                    trace_idx += 1

        print("{:d} traces found.".format(len(det_traces['det_no'])))
        if Globals.is_debug('medium'):
            OptTools._print_det_traces(det_traces, type)
        return det_traces

    @staticmethod
    def _print_det_traces(det_traces, type):
        print('Trace type = ', det_traces['type'])
        fmt = "{:>10s},{:>10s},{:>10s},{:>12s},{:>12s},{:>20s}"
        print(fmt.format('Trace ID', 'Det. no.', 'Slice no.', 'u_mean', 'v_max', 'Rot angle / deg'))
        fmt = "{:10d},{:10d},{:10d},{:12.1f},{:12.1f},{:20.3f}"
        for i in range(len(det_traces['det_no'])):
            trace_idx = det_traces['trace_idx'][i]
            det_no = det_traces['det_no'][i]
            slice_no = det_traces['slice_no'][i]
            mean_v = det_traces['popt'][i][0]
            u_mean = det_traces['u_mean'][i]
            v_max = det_traces['v_max'][i]
            rot_angle = det_traces['rot_angle'][i]
            print(fmt.format(trace_idx, det_no, slice_no, u_mean, v_max, rot_angle))
        return
