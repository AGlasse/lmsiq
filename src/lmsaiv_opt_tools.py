import math
import numpy as np
import scipy
from scipy.optimize import curve_fit, OptimizeWarning
from lms_globals import Globals


class OptTools:


    def __init__(self):
        return

    @staticmethod
    def copy_mosaic(mosaic, clear_data=False, copy_name=''):
        file_name, hdr, hdus = mosaic
        moscopy_hdus = []
        for hdu in hdus:
            moscopy_hdu = hdu.copy()
            if clear_data:
                moscopy_hdu.data *= 0.
            moscopy_hdus.append(moscopy_hdu)
        moscopy_name = file_name if copy_name == '' else copy_name
        moscopy = moscopy_name, hdr, moscopy_hdus
        return moscopy

    # @staticmethod
    # def subtract_mosaics(mos1, mos2):
    #     name1, pri_hdr1, hdu_list1 = mos1
    #     name2, pri_hdr2, hdu_list2 = mos2
    #     mos_hdr, mos_hdus = None, []
    #     for hdu1, hdu2 in zip(hdu_list1, hdu_list2):
    #         hdr = copy.deepcopy(hdu1.header)
    #         hdu = hdu1.copy()
    #         hdu.data = hdu1.data - hdu2.data
    #         mos_hdus.append(hdu)
    #     moscopy_name = name1[:-5] + '_bs.fits'
    #     moscopy = moscopy_name, pri_hdr1, mos_hdus
    #     return moscopy

    # @staticmethod
    # def add_mosaics(mos1, mos2):
    #     name1, pri_hdr1, hdu_list1 = mos1
    #     name2, pri_hdr2, hdu_list2 = mos2
    #     mos_hdr, mos_hdus = None, []
    #     for hdu1, hdu2 in zip(hdu_list1, hdu_list2):
    #         hdr = copy.deepcopy(hdu1.header)
    #         hdu = hdu1.copy()
    #         hdu.data = hdu1.data + hdu2.data
    #         mos_hdus.append(hdu)
    #     moscopy_name = name1[:-5] + '_ca.fits'
    #     moscopy = moscopy_name, pri_hdr1, mos_hdus
    #     return moscopy
    #
    @staticmethod
    def dark_stats(mosaics):
        for mosaic in mosaics:
            file_name, hdr, hdus = mosaic
            dit = hdr['HIERARCH ESO DET DIT']
            ndit = hdr['HIERARCH ESO DET NDIT']
            t_int = dit * ndit

            print()
            print("File = {:s}".format(file_name))
            fmt = "Signal distribution statistics, ndit= {:d} x dit= {:3.1f} sec, integration time = {:6.1f}"
            print(fmt.format(ndit, dit, t_int))
            fmt = "{:>8s},{:>10s},{:>10s},{:>10s},{:>10s}"
            print(fmt.format('Detector', 'median', 'stdev', 'median', 'Rd_Noise'))
            print(fmt.format('No.', 'DN', 'DN', 'el/sec.', 'el.'))
            fmt = "{:8d},{:10.3f},{:10.3f},{:10.3f},{:10.3f}"

            for i, hdu in enumerate(hdus):
                el_adu = float(hdu.header['HIERARCH ESO DET3 CHIP GAIN'])
                median = np.median(hdu.data)
                stdev = np.std(hdu.data)
                median_current = median * el_adu / dit
                rd_noise = stdev * el_adu / math.sqrt(2. / ndit)
                text = fmt.format(i + 1, median, stdev, median_current, rd_noise)
                print(text)
        return

    @staticmethod
    def transform_detector_image(mosaic, det_no, xy_pix=(0, 0), angle=0.0):
        mos_name, mos_primary_header, mos_hdus = mosaic
        det_idx = det_no - 1
        cosa = math.cos(math.radians(angle))
        sina = math.sin(math.radians(angle))
        img = mos_hdus[det_idx].data
        cval = np.median(img)
        tr_mat = np.array([[cosa, sina, 0.], [-sina, cosa, 0.], [0., 0., 1.]])
        rotimg = scipy.ndimage.affine_transform(img, tr_mat, cval=cval, order=3)
        mos_hdus[det_idx].data = rotimg
        return mosaic

    @staticmethod
    def extract_det_traces(mosaic, type, slice_map,
                           snr_cut=100, lt_waves=None):
        """ Extract iso-alpha or iso-lambda traces from a spectral image.
        :param mosaic: Data tuple (name, , image list)  Background subtracted trace mosaic
        :param type: 'alpha' or 'lambda'
        :param slice_map:
        :param kwargs:
        :return:
        """
        popt, pcov = None, None
        mos_name, mos_primary_header, mos_hdus = mosaic
        pri_ang = mos_primary_header['HIERARCH AIT PRI_ANG']
        ech_ang = mos_primary_header['HIERARCH AIT ECH_ANG']
        ref_ech_ord = mos_primary_header['HIERARCH ACHG REF_ECH_ORD']

        is_alpha = type == 'alpha'
        if is_alpha:
            efp_x = mosaic[1]['HIERARCH ACHG WCU X']
            efp_y = mosaic[1]['HIERARCH ACHG WCU Y']

        trace_idx = 0

        det_traces = {'name': mos_name, 'type': type,
                      'pri_ang': pri_ang, 'ech_ang': ech_ang, 'ech_ord': ref_ech_ord,
                      'det_no': [], 'mos_idx': [],
                      'trace_idx': [], 'slice_no': [],
                      'pt_u_coords': [], 'pt_v_coords': [], 'u_mean': [], 'v_max': [],
                      'popt': [], 'pcov': [],
                      'theta': [], 'theta_err': [], 'theta_ufiducial': [],
                      'efp_x': [], 'efp_y': [], 'efp_w': []}
        _, _, slice_map_hdus = slice_map
        fmt = ''
        if Globals.is_debug('low'):
            fmt = "{:>12s},{:>12s},{:>12s},{:>12s},{:>12s},{:>12s},{:>12s}"
            print(fmt.format('Detector', 'Slice ', 'Brightest', 'Signal', 'Bgd', 'Noise', 'SNR'))
            print(fmt.format('Number  ', 'Number', 'Row      ', 'DN/sec', 'DN/sec', '1 sigma', '-'))
            fmt = "{:>12d},{:>12d},{:>12d},{:>12.2f},{:>12.2f},{:>12.2f},{:>12.1f}"

        for hdu in mos_hdus:
            det_no = int(hdu.header['ID'].strip())
            mos_idx = Globals.mos_idx[det_no]
            # Read in image and replace low (dark) values with the median/background level
            raw_image = np.array(hdu.data)
            bgd = np.median(raw_image)
            image = np.where(raw_image < 0.2 * bgd, bgd, raw_image)

            slice_map_data = np.array(slice_map_hdus[mos_idx].data)
            snu = np.unique(slice_map_data)
            slice_nos = [int(s) for s in snu if s > 0.]
            # Start by extracting the data for each slice.
            notrace_slices = []
            for slice_no in slice_nos:
                ism = int(0.5 * Globals.intra_slice_gap)        # Intra-slice margin
                idx = np.argwhere(slice_no == slice_map_data)
                rs_min, rs_max = np.amin(idx[:, 0]) - ism, np.amax(idx[:, 0]) + ism
                u_off = 0 if is_alpha else rs_min
                # Extract slice image
                slice_image = np.array(image[rs_min:rs_max, :])
                u_count = slice_image.shape[1] if is_alpha else slice_image.shape[0]
                v_count = slice_image.shape[0] if is_alpha else slice_image.shape[1]

                # We locate traces by collapsing along rows (for iso-alpha) or columns (for iso-lambda)
                # Notation, u = along trace, v = orthogonal to trace
                axis = 1 if is_alpha else 0
                # Loop to find all traces in this slice image.
                noise = np.median(np.std(slice_image, axis=axis))         # Noise measure for slice image
                bgd = np.median(slice_image)
                v_max_list = []

                repeat = True
                s_aves = np.mean(slice_image, axis=axis)
                while repeat:
                    v_max = np.argmax(s_aves)                           # Brightest row (alpha) / column (lambda)
                    v1_sig, v2_sig = v_max - 5, v_max + 5               # Rows/cols to use for trace analysis
                    v1_clr, v2_clr = v_max - 10, v_max + 10
                    v1_sig, v2_sig = max(0, v1_sig), min(v_count, v2_sig)
                    v1_clr, v2_clr = max(0, v1_clr), min(v_count, v2_clr)
                    signal = np.mean(s_aves[v1_sig:v2_sig])
                    snr = (signal - bgd) / noise
                    repeat = snr > snr_cut
                    if repeat:
                        if Globals.is_debug('low'):
                            text = fmt.format(det_no, slice_no, v_max + rs_min, signal, bgd, noise, snr)
                            print(text)
                        v_max_list.append(v_max)
                        s_aves[v1_clr:v2_clr] = bgd
                if len(v_max_list) < 1:
                    if Globals.is_debug('low'):
                        # print('No traces found for det_no ', det_no, ', slice_no ', slice_no)
                        notrace_slice = "{:d}_{:d}".format(det_no, slice_no)
                        notrace_slices.append(notrace_slice)
                    continue
                else:
                    n_samples = 10 if is_alpha else 5
                    u_interval = u_count // (n_samples + 1)
                    u_start, u_end = u_interval, u_count - u_interval
                    u_list = list(range(u_start, u_end, u_interval))
                    u_hw, v_hw = 5, 10       # Sample half width in along and across trace dimensions to fit gaussian.
                    for v_max in v_max_list:
                        v1, v2 = v_max - v_hw, v_max + v_hw  # Across trace rows/cols to use for trace analysis
                        v1, v2 = max(v1, 0), min(v2, v_count)
                        pt_u_coords, pt_v_coords = [], []
                        for u in u_list:
                            # Find the trace coordinates by fitting gaussians
                            u1, u2 = u - u_hw, u + u_hw
                            u1, u2 = max(u1, 0), min(u2, u_count)
                            sample_image = slice_image[v1:v2, u1:u2] if is_alpha else slice_image[u1:u2, v1:v2]
                            z_vals = np.mean(sample_image, axis=axis)
                            v_vals = np.array(list(range(v1, v2)))
                            idx_max = np.argmax(z_vals)
                            z_max = z_vals[idx_max]
                            v_sigma = 1.0
                            p0_guess = [z_max, v1 + v_hw, v_sigma]
                            try:
                                gopt, gcov = curve_fit(Globals.gauss, v_vals, z_vals, p0=p0_guess)
                                v_off = rs_min if is_alpha else 0
                                v_gauss_peak = gopt[1] + v_off     # Get the row/col coordinate in the image frame.
                                pt_u_coords.append(float(u + u_off))
                                pt_v_coords.append(v_gauss_peak)
                            except:
                                text = fmt.format(det_no, slice_no, v_max + rs_min, signal, bgd, noise, snr)
                                print(text + ' Gaussian fit failed')

                        u_mean = np.mean(pt_u_coords)
                        p0_guess = [u_mean, 0., 0., 0.]
                        n_func_pars = len(p0_guess)
                        n_data_points = len(pt_u_coords)
                        if n_data_points < n_func_pars:
                            continue
                        try:
                            popt, pcov = curve_fit(Globals.polynomial, pt_u_coords, pt_v_coords, p0=p0_guess)
                        except (RuntimeError, OptimizeWarning, ValueError):
                            print('!! Error finding polynomial trace fit !!')
                        det_traces['trace_idx'].append(trace_idx)
                        det_traces['det_no'].append(det_no)
                        det_traces['mos_idx'].append(mos_idx)
                        det_traces['slice_no'].append(slice_no)
                        det_traces['popt'].append(popt)
                        det_traces['pcov'].append(pcov)
                        det_traces['pt_u_coords'].append(pt_u_coords)
                        det_traces['pt_v_coords'].append(pt_v_coords)
                        det_traces['u_mean'].append(u_mean)
                        det_traces['v_max'].append(v_max)
                        if is_alpha:
                            det_traces['efp_x'].append(efp_x)
                            det_traces['efp_y'].append(efp_y)
                        else:
                            nob = 1

                        # Calculate rotation angle as differential of polynomial fit
                        trace_idx += 1
                det_traces['efp_w'] = [None]*trace_idx if is_alpha else lt_waves

            if Globals.is_debug('low'):
                text = "No traces found for det_slice no. = "
                for notrace_slice in notrace_slices:
                    text += notrace_slice + ", "
                print(text)

        det_traces = OptTools._find_thetas(det_traces)
        print("{:d} traces found.".format(len(det_traces['det_no'])))
        if Globals.is_debug('low'):
            OptTools._print_det_traces(det_traces, type)
        return det_traces

    @staticmethod
    def _print_det_traces(det_traces, type):
        print('Trace type = ', det_traces['type'])
        fmt = "{:>10s},{:>10s},{:>10s},{:>12s},{:>12s},{:>20s}"
        print(fmt.format('Trace ID', 'Det. no.', 'Slice no.', 'u_mean', 'v_max', 'theta / deg'))
        fmt = "{:10d},{:10d},{:10d},{:12.1f},{:12.1f},{:20.3f}"
        for i in range(len(det_traces['det_no'])):
            trace_idx = det_traces['trace_idx'][i]
            det_no = det_traces['det_no'][i]
            slice_no = det_traces['slice_no'][i]
            mean_v = det_traces['popt'][i][0]
            u_mean = det_traces['u_mean'][i]
            v_max = det_traces['v_max'][i]
            rot_angle = det_traces['theta'][i]
            print(fmt.format(trace_idx, det_no, slice_no, u_mean, v_max, rot_angle))
        return

    @staticmethod
    def _find_thetas(traces):
        """ Calculate the angle between each trace and the 'u' axis (row for iso-alpha traces, column for iso-lambda),
        evaluated from the gradient of the polynomial fit at location 'u_fiducial'.  The angle and associated
        parameters are written to the trace dictionary as keys beginning 'theta_'
        """
        n_traces = len(traces['trace_idx'])
        traces['theta'] = np.zeros(n_traces)
        traces['theta_err'] = np.zeros(n_traces)
        traces['theta_ufiducial'] = np.zeros(n_traces)
        for idx in traces['trace_idx']:
            popt = traces['popt'][idx]
            pcov = traces['pcov'][idx]
            u_fiducial = traces['u_mean'][idx]
            gradient = Globals.polynomial(u_fiducial, *popt, gradient=True)
            gradient_err = np.sqrt(pcov[1][1])
            deg_rad = 180. / math.pi
            traces['theta'][idx] = deg_rad * math.atan(gradient)
            traces['theta_err'][idx] = gradient_err * deg_rad / (1 + gradient**2)
            traces['theta_ufiducial'][idx] = u_fiducial
        return traces
