import numpy as np
import math
from photutils.centroids import centroid_com
from photutils.aperture import RectangularAperture, CircularAperture
from scipy.optimize import curve_fit


class Analyse:

    fwhm_sigma = None

    def __init__(self):
        Analyse.fwhm_sigma = 2 * math.sqrt(2.0 * math.log(2.0))
        return

    @staticmethod
    def extract_cube_strips(image_list, oversampling):
        first_obs = True
        n_obs = len(image_list)
        c1, c2 = 0, 0
        strips = None
        for i, image in enumerate(image_list):
            if first_obs:
                nzr, nzc = image.shape
                zc_cen = int(nzc / 2)
                spec_width_pix = 2.5  # Spectral width per slice for reconstructed cubes.
                zc_hwid = int(0.5 * spec_width_pix * oversampling)
                c1, c2 = zc_cen - zc_hwid, zc_cen + zc_hwid
                strips = np.zeros((n_obs, nzr))
                first_obs = False
            strip = np.sum(image[:, c1:c2], axis=1)
            strips[i, :] = strip
        return strips

    @staticmethod
    def _gauss(x, amp, fwhm, xpk):
        sigma = fwhm / Analyse.fwhm_sigma
        k = (x - xpk) / sigma
        y = amp * np.exp((-k ** 2) / 2.0)
        return y

    @staticmethod
    def get_dispersion(ds_dict, traces):
        """ Get the mean dispersion at the passed wavelength from the 'trace' objects generated using
        the distortion model.  For now this just works by interpolating the wave v dw_dx_mean values
        for all traces.
        """
        wave = ds_dict['wavelength']

        wave_list, dw_dx_mean_list, dw_dx_std_list = [], [], []
        for trace in traces:
            wave_list.append(trace.mean_wavelength)
            dw_dx_mean, dw_dx_std = trace.dw_dx
            dw_dx_mean_list.append(dw_dx_mean)
            dw_dx_std_list.append(dw_dx_std)
        wave_means = np.array(wave_list)
        sort_indices = np.argsort(wave_means)
        w_sort = wave_means[sort_indices]
        dw_dx_means = np.array(dw_dx_mean_list)[sort_indices]
        dw_dx_stds = np.array(dw_dx_std_list)[sort_indices]
        dw_dx = np.interp(wave, w_sort, dw_dx_means), np.interp(wave, w_sort, dw_dx_stds)
        return dw_dx

    @staticmethod
    def find_fwhm_multi(image_list, **kwargs):

        fits, fit_errs, fwhms_lin, xls, xrs, yhs = [], [], [], [], [], []
        for image in image_list:
            gauss, linear = Analyse.find_fwhm(image, **kwargs)
            is_error_gau, fit, fit_err = gauss
            if is_error_gau:
                continue
            fits.append(fit)
            fit_errs.append(fit_err)

            is_error_lin, fwhm_lin, _, xl, xr, yh = linear
            if is_error_lin:
                continue
            xls.append(xl)
            xrs.append(xr)
            yhs.append(yh)
            fwhms_lin.append(fwhm_lin)

        fits = np.array(fits)
        mean_fit = np.mean(fits, axis=0)
        mean_fit_err = np.std(fits, axis=0)
        gauss = False, mean_fit, mean_fit_err

        xl_mean = np.mean(np.array(xls))
        xr_mean = np.mean(np.array(xrs))
        yh_mean = np.mean(np.array(yhs))
        fwhm_lin_mean = np.mean(np.array(fwhms_lin))
        fwhm_lin_err = np.std(np.array(fwhms_lin))
        linear = False, fwhm_lin_mean, fwhm_lin_err, xl_mean, xr_mean, yh_mean

        return gauss, linear

    @staticmethod
    def find_fwhm(image, **kwargs):
        oversample = kwargs.get('oversample', 1)
        axis = kwargs.get('axis', 0)

        nr5, nc5 = image.shape
        half_aperture = 2 * oversample
        rc_lo = int(nc5 / 2) - half_aperture
        rc_hi = rc_lo + 2 * half_aperture + 1

        n_rows, n_cols = image.shape
        n_pts = n_cols if axis == 0 else n_rows
        sub_image = image[rc_lo:rc_hi] if axis == 0 else image[:, rc_lo:rc_hi]
        y = np.mean(sub_image, axis=axis)
        x = np.arange(0.0, float(n_pts)) / oversample        # Find FWHM in detector units
        is_error_gau, fit, covar = Analyse._fit_gaussian(x, y)
        cv1 = np.array(covar)
        cv2 = np.diag(cv1)
        fit_err = np.sqrt(cv2)
        gauss = is_error_gau, fit, fit_err

        is_error_lin, xl, xr, yh = Analyse._find_fwhm_lin(x, y)
        fwhm_lin = xr - xl
        fwhm_lin_err = 0.
        linear = is_error_lin, fwhm_lin, fwhm_lin_err, xl, xr, yh
        return gauss, linear

    @staticmethod
    def _fit_gaussian(x, y, **kwargs):
        debug = kwargs.get('debug', False)
        imax = np.argmax(y)
        guess_amp, guess_fwhm, guess_xpk = y[imax], 2.0, x[imax]
        guess = [guess_amp, guess_fwhm, guess_xpk]
        is_error = False
        try:
            fit, covar = curve_fit(Analyse._gauss, x, y,
                                   p0=guess, method='trf', xtol=1E-8)
        except RuntimeError:
            print('lmsiq_analyse.fit_gaussian !! Runtime error !!')
            is_error = True
            order = len(guess) + 1
            fit, covar = np.zeros(order), np.zeros((order, order))
        if debug:
            fmt = "{:<8s} - amp={:8.6f} fwhm={:5.2f} xpk={:9.6f}"
            print(fmt.format('Guess', guess_amp, guess_fwhm, guess_xpk))
            print(fmt.format('Fit  ', fit[0], fit[1], fit[2]))
            print()
        return is_error, fit, covar

    @staticmethod
    def _find_fwhm_lin(x, y):
        """ Find the FWMH of a profile by using linear interpolation to find where it crosses the half-power
        level.  Referred to in summary files as the 'linear' FWHM etc.
        """
        ymax = np.amax(y)
        yh = ymax / 2.0
        yz = np.subtract(y, yh)       # Search for zero crossing
        iz = np.where(yz > 0)
        il = iz[0][0]
        ir = iz[0][-1]
        xl, xr, yh = 0., 0., 0.
        is_error = False
        try:
            xl = x[il-1] - yz[il-1] * (x[il] - x[il-1]) / (yz[il] - yz[il-1])
            xr = x[ir] - yz[ir] * (x[ir+1] - x[ir]) / (yz[ir+1] - yz[ir])
        except IndexError:
            is_error = True
        return is_error, xl, xr, yh

    @staticmethod
    def find_phases(data_id, centroids):
        _, _, _, _, _, ndet_cols = data_id
        phase_ref = ndet_cols / 2.0
        _, n_mcruns = centroids.shape

        phases, phase_diffs = np.zeros(centroids.shape), np.zeros(centroids.shape)
        for col in range(1, n_mcruns):
            phases[:, col] = centroids[:, col] - centroids[:, 0] - phase_ref
        phase_diffs[:-1, 1:] = phases[1:, 1:] - phases[:-1, 1:]
        phase_diffs[:, 0], phases[:, 0] = centroids[:, 0], centroids[:, 0]
        return phases, np.abs(phase_diffs)

    @staticmethod
    def find_phot(image, **kwargs):
        method = kwargs.get('method', 'fixed_aperture')
        method_parameters = kwargs.get('method_parameters', None)
        phot = 0.
        if method == 'aperture':
            if method_parameters is not None:
                ap_pos, width, height = method_parameters
                rect_aper = RectangularAperture(ap_pos, w=width, h=height)
                phot = Analyse.exact_rectangular(image, rect_aper)
            else:
                print("!! Analyse.find_phot aperture parameter not set !!")
        if method == 'full_image':
            phot = np.sum(image)
        return phot

    @staticmethod
    def subtract_phases(values):
        _, n_runs = values.shape
        for col in range(1, n_runs):
            values[:, col] = values[:, col] - values[:, 0]
        return values

    @staticmethod
    def find_stats(data_list):
        n_configs = len(data_list)
        stats = np.zeros((n_configs, 3))
        for i, data in enumerate(data_list):
            data_id, zemax_configuration, centroids, photometry, centroid_abs_diffs = data
            wave = zemax_configuration[1]
            abs_diffs = centroid_abs_diffs[1:12, 1:]
            mean_abs_diff = np.mean(abs_diffs)
            mean_abs_diff_err = np.std(abs_diffs)
            stats[i, :] = [wave, mean_abs_diff, mean_abs_diff_err]
        return stats

    @staticmethod
    def fix_offset(centroids):
        _, n_cols = centroids.shape
        for run in range(1, n_cols):
            mean_offset = np.mean(centroids[3:20, run]) - np.mean(centroids[23:40, run])
            centroids[3:20, run] -= mean_offset
        return centroids

    @staticmethod
    def find_centroid(image):
        centroid = centroid_com(image)
        return centroid

    @staticmethod
    def _find_mean_centroid(images):
        """ Find the mean centroid position for all images in a list.
        """
        cube = np.array(images)
        image_ave = np.average(cube, axis=0)
        centroid = centroid_com(image_ave)
        return centroid

    @staticmethod
    def find_phase_shift_rates(centroids):
        nrows, ncols = centroids.shape
        phase_shift_rates = np.zeros(centroids.shape)
        phase_shift_rates[:, 0] = centroids[:, 0]
        for i in range(1, nrows):
            for j in range(1, ncols):
                psr = (centroids[i, j] - centroids[i-1, j]) / (centroids[i, 0] - centroids[i-1, 0])
                phase_shift_rates[i, j] = psr
        return phase_shift_rates

    @staticmethod
    def find_strehls(cube_stack):
        n_runs, n_rows, n_cols = cube_stack.shape
        strehls = np.zeros(n_runs)
        p_factor = None
        rows = np.arange(0, n_rows)
        first_time = True
        for run_idx in range(0, n_runs):
            image = cube_stack[run_idx]
            isum = np.sum(image)
            rmax, cmax = np.unravel_index(np.argmax(image, keepdims=True), image.shape)
            x = rows
            y = np.squeeze(image[:, cmax])
            _, gauss_fit, _ = Analyse._fit_gaussian(x, y)
            imax = gauss_fit[0]
            if first_time:
                psum = isum
                pmax = imax
                p_factor = psum / pmax
                first_time = False
            strehl = (imax / isum) * p_factor
            # print("{:03d}, {:8.5f}, {:8.5f}, {:6.3f}".format(run_idx, imax, isum, strehl))
            strehls[run_idx] = strehl
        return strehls

    @staticmethod
    def eed(image_list, obs_dict, axis_name, **kwargs):
        """ Calculate the enslitted energy along an axis.
        :param image_list:              List of imagess
        :param obs_dict            List of parameters associated with image list
        :param axis_name:                    ='spectral' or 'spatial'
        :param kwargs: is_log = True for samples which are uniform in log space
        :return radii: Sampled axis
                ees_mean: Mean enslitted energy profile
                ees_rms:
                ees_all: EE profile averaged for all images
        """
        debug = kwargs.get('debug', False)
        is_log = kwargs.get('log10sampling', True)  # Use sampling equispaced in log10
        oversample = kwargs.get('oversample', 1.)

        ny, nx = image_list[0].shape
        nxy_min = nx if nx < ny else ny

        n_obs = len(image_list)
        r_sample = 0.1
        r_start = r_sample
        # Set maximum radial size of aperture (a bit less than the full image size to avoid OOB errors)
        r_max = (nxy_min / 2.) - 1.

        radii = np.arange(r_start, r_max, r_sample)
        n_points = radii.shape[0]
        if is_log:      # Remap the same number of points onto a log uniform scale.
            k = math.log10(r_max / r_start) / (n_points - 1)
            lr = math.log10(r_start)
            for i in range(0, n_points):
                radii[i] = math.pow(10., lr)
                lr += k

        ees_all = np.zeros((n_points, n_obs))
        im_sums = np.zeros(n_obs)
        im_peaks = np.zeros(n_obs)
        if debug:
            print()

        for j, image in enumerate(image_list):
            file_name = obs_dict['file_names'][j]
            img_min, img_max = np.amin(image), np.amax(image)
            centroid = centroid_com(image)
            umin, vmin = centroid[0] - r_max, centroid[1] - r_max
            umax, vmax = centroid[0] + r_max, centroid[1] + r_max
            is_x_oob = umin < 0 or umax > nx - 1
            is_y_oob = vmin < 0 or vmax > ny - 1
            if debug:
                fmt = "\r- processing file {:s} into column {:d} {:10.1e} {:10.1e}"
                print(fmt.format(file_name, j, img_min, img_max))
                if is_x_oob or is_y_oob:
                    txt = "!! U/V out of bounds (signal truncated) !! - "
                    fmt = "u={:5.1f} -{:5.1f}, v={:5.1f} -{:5.1f}"
                    txt += fmt.format(umin, umax, vmin, vmax)
                    print(txt)

            for i in range(0, n_points):        # One radial point per row
                r = radii[i]      # Increase aperture width to measure spectral cog profile
                aperture = None
                if axis_name == 'radial':
                    aperture = CircularAperture(centroid, r)
                    signal_list = aperture.do_photometry(image)
                    signal = signal_list[0][0]
                else:
                    if axis_name in ['spectral', 'across-slice']:
                        aperture = RectangularAperture(centroid, w=2.*r, h=2.*r_max)     # Spectral
                    if axis_name in ['spatial', 'along-slice']:
                        aperture = RectangularAperture(centroid, w=2.*r_max, h=2.*r)     # Spatial
                    signal = Analyse.exact_rectangular(image, aperture)
                ees_all[i, j] = signal

            enorm = np.sum(image)
            ees_all[:, j] = np.divide(ees_all[:, j], enorm)
            im_sums[j] = ees_all[n_points-1, j]
            im_peaks[j] = img_max

        # Rescale x axis from image scale to LMS pixels
        # ees_perfect, ees_design = ees_all[:, 0], ees_all[:, 1]
        ees_mcs = ees_all[:, 2:]
        ees_mc_mean = np.mean(ees_mcs, axis=1)
        ees_mc_rms = np.std(ees_mcs, axis=1)
        xdet = np.divide(radii, oversample)         # Convert x scale to LMS detector pixels
        ees_data = {'xvals': xdet, 'yvals': ees_all,
                    'ees_mc_mean': ees_mc_mean, 'ees_mc_rms': ees_mc_rms}
        return ees_data

    @staticmethod
    def find_ee_axis_references(wav, ee_data):
        """ Calculate the EED for the three EE profiles (mean, perfect and design) at a reference aperture.
        """
        axis, x, ee_per, ee_des, ee_mc, ee_mc_err, ee_mc_all = ee_data

        axis_tag = '_' + axis[0:4]
        ee_tag = 'ee' + axis_tag + '_'

        x_ref_min = 2.
        broadening = 1.0 if wav < 3.7 else wav / 3.7
        x_ref = x_ref_min * broadening

        ees = {ee_tag + 'per': ee_per,
               ee_tag + 'des': ee_des,
               ee_tag + 'mean': ee_mc}

        ee_refs = {ee_tag + 'ref': x_ref}
        iz = np.where(x > x_ref)
        is_index = len(iz[0])
        i = iz[0][0] if is_index else len(x) - 1
        for key in ees:
            ee = ees[key]
            ee_val = ee[i - 1] + (x_ref - x[i - 1]) * (ee[i] - ee[i - 1]) / (x[i] - x[i - 1])
            ee_refs[key] = ee_val
        return x_ref, ee_refs

    @staticmethod
    def ycompress(cube_in, factor):
        n_layers, n_obs, n_rows_in, n_cols = cube_in.shape
        n_rows_out = int(n_rows_in / factor) + 1
        shape_out = n_layers, n_obs, n_rows_out, n_cols
        cube_out = np.zeros(shape_out)
        ap_width = 1.
        for i in range(0, n_layers):
            for j in range(0, n_obs):
                image = cube_in[i, j, :, :]
                for col in range(0, n_cols):
                    ap_centre_col = col + .5
                    for row_out in range(0, n_rows_out):
                        ap_centre_row = factor * (row_out + 0.5)
                        ap_centre = ap_centre_col, ap_centre_row
                        ap_height = factor
                        aperture = RectangularAperture(ap_centre, w=ap_width, h=ap_height)
                        cube_out[i, j, row_out, col] = Analyse.exact_rectangular(image, aperture)
        return cube_out

    # @staticmethod
    # def strehl(observations):
    #     """ Calculate the Strehl ratio as the ratio between the peak amplitude of the mean image and
    #     the peak amplitude of the perfect image, where both images have been normalised to have a total
    #     signal of unity
    #     """
    #     perfect_image, _ = observations[0]
    #     perfect_flux = np.sum(perfect_image)
    #     perfect_peak = np.amax(perfect_image)
    #     # Calculate the error on the Strehl from the individual images
    #     strehl_list = []
    #     for obs in observations:
    #         image, params = obs
    #         power = np.sum(image)
    #         peak = np.amax(image)
    #         strehl = (peak * perfect_flux) / (perfect_peak * power)
    #         strehl_list.append(strehl)
    #     strehl_err = np.std(np.array(strehl_list))
    #     strehl_mean = np.mean(np.array(strehl_list))
    #     return strehl_mean, strehl_err

    @staticmethod
    def _boxcar(series_in, box_width):
        n_pts, _ = series_in.shape
        series_out = np.zeros(series_in.shape)
        box_hw = int(box_width / 2.)
        for i in range(0, n_pts):
            i1, i2 = i - box_hw, i + box_hw
            i1 = i1 if i1 >= 0 else 0
            i2 = i2 if i2 < n_pts else n_pts
            series_out[i, :] = np.mean(series_in[i1:i2, :], axis=0)
        return series_out

    @staticmethod
    def lsf(image_list, obs_dict, axis, **kwargs):
        """ Find the normalised line spread function for all image files.  Note that the pixel is centred
        at its index number, so we perform rectangular aperture photometry centred at
        :returns - uvals, pixel scale
                 - lsf_mean, mean line spread function for all images
                 - lsf_rms, root mean square distribution per pixel
                 - lsf, array of line spread functions for each image
        """
        debug = kwargs.get('debug', False)
        centroid_relative = kwargs.get('centroid_relative', True)
        oversample = kwargs.get('oversample', 1.)   # Pixel oversampling with respect to detector
        boxcar = kwargs.get('boxcar', False)        # t=Apply boxcar filter of width = oversample
        v_coadd = kwargs.get('v_coadd', 'all')      # Number of image pixels to coadd orthogonal to profile
        u_radius = kwargs.get('u_radius', 'all')    # Maximum radial size of aperture (a bit less than 1/2 image size)
        usample = 1.0                               # Sample psf once per pixel to avoid steps
        ustart = 0.0                                # Offset from centroid

        n_rows, n_cols = image_list[0].shape

        vcoadd = n_cols - 1 if v_coadd == 'all' else v_coadd
        uradius = (n_rows - 1) / 2.0 if u_radius == 'all' else u_radius
        if axis == 0:
            vcoadd = n_rows - 1 if v_coadd == 'all' else v_coadd
            uradius = (n_cols - 1) / 2.0 if u_radius == 'all' else u_radius

        uvals = np.arange(ustart - uradius, ustart + uradius + 1., usample)
        n_points = uvals.shape[0]

        n_files = len(image_list)
        lsf_all = np.zeros((n_points, n_files))

        centroid = np.zeros((2,))
        if centroid_relative:
            centroid = Analyse._find_mean_centroid(image_list)

        for j, image in enumerate(image_list):
            file_name = obs_dict['file_names'][j]
            if debug:
                print('Processing file {:s} into column {:d}'.format(file_name, j))
            ap_pos = np.array(centroid)
            ucen = centroid[0] if axis == 0 else centroid[1]
            us = np.add(uvals, ucen)
            for i in range(0, n_points):  # One radial point per row
                u = us[i]
                if axis == 0:       # spectral or across-slice
                    ap_pos[0] = u
                    aperture = RectangularAperture(ap_pos, w=usample, h=vcoadd)
                    lsf_all[i, j] = Analyse.exact_rectangular(image, aperture)
                if axis == 1:       # Spatial or along-slice
                    ap_pos[1] = u
                    aperture = RectangularAperture(ap_pos, w=vcoadd, h=usample)  # Spatial
                    lsf_all[i, j] = Analyse.exact_rectangular(image, aperture)
        lsf_norm = lsf_all / np.amax(lsf_all, axis=0)
        xvals = np.divide(uvals, oversample)         # Convert x scale to LMS detector pixels
        if boxcar:
            lsf_norm = Analyse._boxcar(lsf_norm, oversample)

        lsf_data = {'xvals': xvals, 'yvals': lsf_norm}

        mc_start, _ = obs_dict['mc_bounds']
        pd_tags, model_tags = ['perfect', 'design'], []
        lsf_data['lin_fwhm'], lsf_data['lin_xl'], lsf_data['lin_xr'] = [], [], []
        lsf_data['gau_fwhm'], lsf_data['gau_xl'], lsf_data['gau_xr'], lsf_data['gau_amp'] = [], [], [], []

        for i, image in enumerate(image_list):
            key = "MC_{:03d}".format(mc_start + i) if i > 1 else pd_tags[i]
            model_tags.append(key)

            gauss, linear = Analyse.find_fwhm(image, axis=axis, debug=True)
            _, fwhm_per_lin, _, xl_lin, xr_lin, _ = linear
            _, (amp_gau, fwhm_per_gau, xcen_gau), covar = gauss
            xl_gau = xcen_gau - 0.5 * fwhm_per_gau
            xr_gau = xcen_gau + 0.5 * fwhm_per_gau
            # Scale to detector pixels and write to data dictionary
            lsf_data['lin_fwhm'].append(fwhm_per_lin / oversample)
            lsf_data['lin_xl'].append(xl_lin / oversample)
            lsf_data['lin_xr'].append(xr_lin / oversample)
            lsf_data['gau_fwhm'].append(fwhm_per_gau / oversample)
            lsf_data['gau_xl'].append(xl_gau / oversample)
            lsf_data['gau_xr'].append(xr_gau / oversample)
            lsf_data['gau_amp'].append(amp_gau)

        return lsf_data

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
    def exact_rectangular(image, aperture):
        """ Calculate the signal within an aperture which may be non-integer in pixel shape.  Note that
        it assumes that pixel index i,j extends from i-pco to i+pco etc.  Here, pco is the pixel centre offset
        relative to the index value.
        """
        pco = 0.5       # Pixel centre offset,

        cen = aperture.positions
        w = aperture.w
        h = aperture.h
        x1 = cen[0] - (w / 2.0) + pco
        x2 = x1 + w
        y1 = cen[1] - (h / 2.0) + pco
        y2 = y1 + h
        c1, c2, r1, r2 = int(x1), int(x2), int(y1), int(y2)
        c1, r1 = max(0, c1),max(0, r1)
        rmax, cmax = image.shape
        c2, r2 = min(cmax-1, c2), min(rmax-1, r2)
        # Number of rows and columns in subarray, 1 pixel extra to allow sub-pixel fragments.
        nr = r2 - r1 + 1
        nc = c2 - c1 + 1
        wts = np.ones((nr, nc))
        im = image[r1:r1+nr, c1:c1+nc]

        fc1 = 1. - (x1 - c1)
        fc2 = x2 - c2
        if nc == 1:
            fc = fc1 + fc2 - 1.
            wts[:, 0] *= fc
        else:
            wts[:, 0] *= fc1
            wts[:, nc-1] *= fc2

        fr1 = 1. - (y1 - r1)
        fr2 = y2 - r2
        if nr == 1:
            fr = fr1 + fr2 - 1.
            wts[0, :] *= fr
        else:
            wts[0, :] *= fr1
            wts[nr-1, :] *= fr2

        wtim = np.multiply(im, wts)
        f = np.sum(wtim)
        return f
