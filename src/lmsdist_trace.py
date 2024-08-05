#!/usr/bin/env python
"""
@author Alistair Glasse
Python object to encapsulate ray trace data for the LMS

18/12/18  Created class (Glasse)
"""
import math
import numpy as np
import matplotlib.patches as mpatches
from scipy.optimize import least_squares
from lms_detector import Detector
from lms_globals import Globals
from lms_util import Util
from lmsdist_plot import Plot


class Trace:

    common_series_names = ['ech_order', 'slice', 'spifu_slice', 'wavelength',
                           'slicer_x', 'slicer_y', 'ifu_x', 'ifu_y']
    nominal_series_names = ['fp2_y', 'slit_x', 'slit_y']
    spifu_series_names = ['fp1_y', 'sp_slicer_x', 'sp_slicer_y']
    nominal_focal_planes = {'LMS EFP': ('fp2_x', 'fp2_y'), 'Slicer': ('slicer_x', 'slicer_y'),
                            'IFU': ('ifu_x', 'ifu_y'), 'Slit': ('slit_x', 'slit_y'),
                            'Detector': ('det_x', 'det_y')}
    spifu_focal_planes = {'LMS EFP': ('fp1_x', 'fp1_y'), 'Slicer': ('slicer_x', 'slicer_y'),
                          'IFU': ('ifu_x', 'ifu_y'), 'SP slicer': ('sp_slicer_x', 'sp_slicer_y'),
                          'Detector': ('det_x', 'det_y')}

    is_spifu = None
    parameter = None
    n_slices, n_rays_slice, n_spifu_slices, n_rays_spifu_slice = None, None, None, None
    unique_slices, unique_spifu_slices, unique_ech_orders = [None], [None], [None]
    coord_in, coord_out = None, None
    mean_wavelength = 0.0

    def __init__(self, path, coord_in, coord_out, **kwargs):
        """ Read in Zemax ray trace data for a specific LMS configuration (echelle and prism angle, spectral IFU
        selected or not).
        @author: Alistair Glasse
        Read Zemax ray trace data
        """
        self.slice_objects, self.ray_trace = None, None
        self.offset_data, self.wave_bounds = None, None
        self.coord_in = coord_in
        self.coord_out = coord_out
        is_spifu = kwargs.get('is_spifu', False)
        self.is_spifu = is_spifu
        silent = kwargs.get('silent', False)
        if not silent:
            print('Reading Zemax model data from ' + path)

        csv_name, parameter, series = self._read_csv(path, is_spifu, coord_in, coord_out)
        self.parameter = parameter
        # For SPIFU data, spifu slices 1,2 -> echelle order 23, 3,4 _> 24, 5,6 -> 25
        if is_spifu:
            spifu_slices = series['spifu_slice']
            spifu_indices = spifu_slices - 1
            ech_orders = np.mod(spifu_indices, 3) + 23
            series['ech_order'] = ech_orders

        waves = series['wavelength']
        ech_orders = series['ech_order']
        d = Globals.rule_spacing * Globals.ge_refracive_index
        sin_aoi = waves * ech_orders / (2.0 * d)
        phase = np.arcsin(sin_aoi)
        series['phase'] = phase         # + phase_corr
        self.series = series

        # Count the number of slices and rays
        self.unique_ech_orders = np.unique(ech_orders)
        slices = series['slice']
        self.unique_slices = np.unique(slices)
        spifu_slices = series['spifu_slice']
        self.unique_spifu_slices = np.unique(spifu_slices)
        self.n_rays, = slices.shape

        self._update_mask(silent)
        self._create_wave_colours()
        self._find_dispersion()
        return

    def __str__(self):
        fmt = "EA={:6.3f} deg, PA={:6.3f} deg, "
        string = fmt.format(self.parameter['Echelle angle'], self.parameter['Prism angle'])
        string += '+SpIFU, ' if self.is_spifu else ''
        smin, smax = self.unique_slices[0], self.unique_slices[-1]
        string += "slices {:d}-{:d}".format(int(smin), int(smax))
        return string

    @staticmethod
    def waves_to_phase(waves, ech_orders):
        d = Globals.rule_spacing * Globals.ge_refracive_index
        sin_aoi = waves * ech_orders / (2.0 * d)
        phase = np.arcsin(sin_aoi)
        return phase

    def _find_dispersion(self, **kwargs):
        """ Calculate the dispersion relation (nm/pixel) as a linear fit to
        all wavelength and det_x data.
        """
        do_plot = kwargs.get('do_plot', False)

        waves = self.series['wavelength']
        det_x = self.series['det_x']
        det_y = self.series['det_y']
        sorted_indices = np.argsort(waves)
        w_sort, x_sort, y_sort = waves[sorted_indices], det_x[sorted_indices], det_y[sorted_indices]
        dw_sort = w_sort[1:] - w_sort[:-1]
        nz_indices = np.nonzero(dw_sort)
        w = 1000. * w_sort[nz_indices]              # Convert wavelength units from micron to nm
        x, y = x_sort[nz_indices], y_sort[nz_indices]
        dw, dx, dy = w[1:] - w[:-1], x[1:] - x[:-1], y[1:] - y[:-1]
        dw_dx, dw_dy = -dw / dx, -dw / dy
        dw_dx_mean, dw_dx_std = np.mean(dw_dx), np.std(dw_dx)
        title = "Dispersion [nm / mm] = {:6.3f} $\pm$ {:5.3f}".format(dw_dx_mean, dw_dx_std)
        if do_plot:
            self.plot_dispersion(x[1:], y[1:], dw_dx, title=title)
        self.dw_dx = dw_dx_mean, dw_dx_std

        return

    def _read_csv(self, path, is_spifu, coord_in, coord_out):
        """ Read trace data in from csv file pointed to by path
        :return:
        """
        csv_name = path.split('/')[-1]
        name = csv_name.split('.')[0]
        with open(path, 'r') as text_file:
            read_data = text_file.read()
        line_list = read_data.split('\n')
        line_iter = iter(line_list)

        # Read configuration parameters
        parameter = {'name': name}
        while True:
            line = next(line_iter)
            tokens = line.split(':')
            if len(tokens) < 2:
                break
            name, val = tokens[0], float(tokens[1])
            parameter[name] = val
            if is_spifu:            # For SPIFU data, the order is currently calculated from the spifu slice number.
                parameter['Spectral order'] = -1
            self.parameter = parameter

        # Create dictionary of data series, include echelle order, slice, spifu_slice, wavelength, and input and output
        # focal planes.
        line = next(line_iter)
        tokens = line.split(',')
        zem_column = {}
        for i, token in enumerate(tokens):
            tag = token.strip()
            zem_column[tag] = i

        fp_names = [coord_in[0], coord_in[1], coord_out[0], coord_out[1]]
        series_names = Trace.nominal_series_names + Trace.common_series_names
        if is_spifu:
            series_names = Trace.spifu_series_names + Trace.common_series_names
        series_names += fp_names
        series = {}
        for name in series_names:
            series[name] = []
        while True:
            line = next(line_iter)
            tokens = line.split(',')
            if len(tokens) < 2:
                break
            for name in series_names:
                if name == 'phase':             # Add the phase column later
                    val = 0.
                else:
                    if name == 'ech_order':         # Assign spectral IFU orders later
                        val = 0 if is_spifu else parameter['Spectral order']
                    else:
                        if name == 'spifu_slice':
                            val = -1       # Default for nominal configuration
                            if is_spifu:
                                zem_col = zem_column['sp_slice']
                                val = int(tokens[zem_col])
                        else:
                            zem_col = zem_column[name]
                            val = float(tokens[zem_col])
                series[name].append(val)
        # Convert series lists to numpy arrays
        int_arrays = ['ech_order', 'slice', 'spifu_slice']
        for name in series:
            vals = series[name]
            series[name] = np.array(vals, dtype=int) if name in int_arrays else np.array(vals)
        return csv_name, parameter, series

    def _update_mask(self, silent):
        edge = Detector.pix_edge * Detector.det_pix_size / 1000.0
        xy_f = (Globals.det_gap / 2.0) + edge - Globals.margin
        xy_n = (Globals.det_gap / 2.0) + Globals.margin
        xy_bounds = {'BL': [-xy_f, -xy_n, -xy_f, -xy_n], 'TL': [-xy_f, -xy_n,  xy_n,  xy_f],
                     'BR': [xy_n,  xy_f, -xy_f, -xy_n], 'TR': [xy_n,  xy_f,  xy_n,  xy_f]}
        self.xy_bounds = xy_bounds
        x = self.series['det_x']
        y = self.series['det_y']

        n_pts = len(x)
        mask = np.zeros(n_pts, dtype='int')	    # >0 = Ray hits a detector
        for det_no, det_name in enumerate(xy_bounds):               # Loop through the 4 detectors
            xy_b = xy_bounds[det_name]
            idx_gt_xlo = x > xy_b[0]
            idx_lt_xhi = x < xy_b[1]
            idx_in_x = np.logical_and(idx_gt_xlo, idx_lt_xhi)
            idx_gt_ylo = y > xy_b[2]
            idx_lt_yhi = y < xy_b[3]
            idx_in_y = np.logical_and(idx_gt_ylo, idx_lt_yhi)
            idx_in = np.where(np.logical_and(idx_in_x, idx_in_y))
            mask[idx_in] = det_no + 1
        n_hits = np.count_nonzero(mask)
        if not silent:
            fmt = 'Rays hitting any detector = {:10d} / {:10d}'
            print(fmt.format(n_hits, n_pts))
        self.mask = mask
        return

    def _create_wave_colours(self):
        wav = self.series['wavelength']
        slices = self.series['slice']
        n_pts = len(wav)
        rgb = np.zeros((3, n_pts))

        w_min = np.amin(wav)
        w_max = np.amax(wav)
        r_min = 0.0
        r_max = 1.0
        b_min = 1.0
        b_max = 0.0

        s_min = int(min(slices))
        s_max = int(max(slices))
        n_slices = s_max - s_min + 1
        self.w_slice = np.zeros((4, n_slices))
        self.w_slice[0, :] = 100.
        self.w_slice[1, :] = -100.
        self.w_slice[2, :] = 100.
        self.w_slice[3, :] = -100.
        w = 0.
        for i in range(0, n_pts):
            w = wav[i]
            s = int(slices[i]) - s_min
            if self.mask[i] > 0:
                d = self.mask[i]
                if d == 1 or d == 2:
                    self.w_slice[0, s] = w if w < self.w_slice[0, s] else self.w_slice[0, s]
                    self.w_slice[1, s] = w if w > self.w_slice[1, s] else self.w_slice[1, s]
                else:
                    self.w_slice[2, s] = w if w < self.w_slice[2, s] else self.w_slice[2, s]
                    self.w_slice[3, s] = w if w > self.w_slice[3, s] else self.w_slice[3, s]

                f = (w - w_min) / (w_max - w_min)
                r = r_min + f * (r_max - r_min)
                g = math.sin(f * math.pi)
                b = b_min + f * (b_max - b_min)
            else:						# Not detected - grey out.
                r = 0.3
                g = 0.3
                b = 0.3
            rgb[0, i] = r
            rgb[1, i] = g
            rgb[2, i] = b
        self.mean_wavelength = np.mean(w)
        self.rgb = rgb
        return

    def _get_parameters(self):
        ea = self.parameter['Echelle angle']
        pa = self.parameter['Prism angle']
        w1 = np.max(self.w_slice[0, :])
        w2 = np.min(self.w_slice[1, :])
        w3 = np.max(self.w_slice[2, :])
        w4 = np.min(self.w_slice[3, :])
        return ea, pa, w1, w2, w3, w4

    @staticmethod
    def _find_limits(a, margin):
        amin = min(a)
        amax = max(a)
        arange = amax - amin
        amargin = margin * arange
        limits = [amin - amargin, amax + amargin]
        return limits

    def get(self, tag, **kwargs):
        """ Extract a specific coordinate (identified by 'tag') from the trace for rays which pass through
        a specified spatial and (if the spectral IFU is selected) spectral slice.
        """
        slice_no = kwargs.get('slice_no', None)
        spifu_no = kwargs.get('spifu_no', None)      # -1 = No Spectral IFU slice set
        a = self.series[tag]
        if slice_no is not None:
            slices = self.series['slice']
            idx = slices == slice_no
            if self.is_spifu:
                spifu_slices = self.series['spifu_slice']
                idx2 = spifu_slices == spifu_no
                idx = np.logical_and(idx, idx2)
            a = np.compress(idx, a)
        return a

    @staticmethod
    def _sort(ref, unsort):
        """ Re-order arrays in list 'unsort' using the index list from sorting array 'ref' in ascending order.
        """
        indices = np.argsort(ref)
        sort = []
        for array in unsort:
            sort.append(array[indices])
        return ref[indices], tuple(sort)

    @staticmethod
    def offset_error_function(p, x, y):
        """ Polynomial with odd powered terms.
        """
        residual = (x * (p[0] + x * x * (p[1] + p[2] * x * x))) - y
        return residual

    @staticmethod
    def _sine_fit(x_ref, x_in, y_in):
        _, xy_sort = Trace._sort(x_ref, (x_in, y_in))
        guess_oc = [0.0, 0.0, 0.0]
        res_lsq = least_squares(Trace.offset_error_function, guess_oc, args=xy_sort)
        offset_correction = res_lsq.x
        # fmt = "p_in={:10.2e},{:10.2e},{:10.2e} -> p_out={:10.2e},{:10.2e},{:10.2e}"
        # print(fmt.format(guess_oc[0], guess_oc[1], guess_oc[2],
        #                  offset_correction[0], offset_correction[1], offset_correction[2]))
        return offset_correction

    def create_transforms(self, n_terms, **kwargs):
        """ Generate transforms to map from phase, across-slice coordinates in the entrance focal plane, to detector
        row, column coordinates.  Also update the wavelength values at the centre and corners of the detector mosaic.
        """
        debug = kwargs.get('debug', False)
        self.slice_objects = []
        off_x2_fits, off_y2_fits = [], []
        offset_data = []
        slice_order = n_terms - 1
        fp_in, fp_out = self.coord_in, self.coord_out
        is_first = True
        for ech_order in self.unique_ech_orders:
            for spifu_no in self.unique_spifu_slices:
                for slice_no in self.unique_slices:
                    waves = self.get('wavelength', slice_no=slice_no, spifu_no=spifu_no)
                    phase = self.get(fp_in[0], slice_no=slice_no, spifu_no=spifu_no)
                    alpha = self.get(fp_in[1], slice_no=slice_no, spifu_no=spifu_no)
                    det_x = self.get(fp_out[0], slice_no=slice_no, spifu_no=spifu_no)
                    det_y = self.get(fp_out[1], slice_no=slice_no, spifu_no=spifu_no)

                    a, b = Util.distortion_fit(phase, alpha, det_x, det_y, slice_order, inverse=False)
                    ai, bi = Util.distortion_fit(phase, alpha, det_x, det_y, slice_order, inverse=True)
                    det_x_fit1, det_y_fit1 = Trace.apply_polynomial_distortion(phase, alpha, a, b)

                    off_x1, off_y1 = det_x - det_x_fit1, det_y - det_y_fit1
                    x_sine_corr = self._sine_fit(det_x_fit1, det_x_fit1, off_x1)
                    y_sine_corr = self._sine_fit(det_x_fit1, det_x_fit1, off_y1)
                    det_x_fit2 = Trace.apply_sine_correction(x_sine_corr, det_x_fit1, det_x_fit1, True)
                    det_y_fit2 = Trace.apply_sine_correction(y_sine_corr, det_x_fit1, det_y_fit1, True)

                    off_x2, off_y2 = det_x - det_x_fit2, det_y - det_y_fit2
                    off_x2_fits.append(off_x2)
                    off_y2_fits.append(off_y2)

                    config = ech_order, slice_no, spifu_no
                    matrices = a, b, ai, bi
                    offset_corrections = x_sine_corr, y_sine_corr
                    rays1 = waves, phase, alpha, det_x, det_y, det_x_fit1, det_y_fit1
                    tf_inter = config, matrices, offset_corrections, rays1
                    rays2 = waves, phase, alpha, det_x, det_y, det_x_fit2, det_y_fit2
                    slice_object = config, matrices, offset_corrections, rays2
                    self.slice_objects.append(slice_object)
                    if debug and is_first:        # Plot intermediate and full fit to data
                        tlin1 = "Distortion offsets, A,B, polynomial fit \n"
                        self.plot_scatter(tf_inter, plot_correction=True, tlin1=tlin1)
                        tlin1 = "Distortion offsets, post-sinusoidal correction \n"
                        self.plot_scatter(slice_object, tlin1=tlin1)
                        is_first = False
        self.offset_data = offset_data
        return

    def add_wave_bounds(self, **kwargs):
        """ Calculate the wavelength bounds using the detector x dimensions and the mean y coordinate for slices
        0, 14 and 27 as input for the transforms to find the phase, alpha coordinates.
        """
        debug = kwargs.get('debug', False)

        xy_bounds = self.xy_bounds
        xy_bl, xy_br = xy_bounds['BL'], xy_bounds['BR']
        det_x_bounds = [xy_bl[0], xy_bl[1], xy_br[0], xy_br[1]]
        tfw_list = []
        wb_list = []
        all_det_x_bounds, all_det_x_rtn, all_det_y_bounds, all_det_y_rtn = [], [], [], []
        for tf in self.slice_objects:
            config, matrices, offset_correction, rays = tf
            ech_order, slice_no, spifu_no = config
            a, b, ai, bi = matrices
            _, _, _, _, det_y, _, _ = rays
            det_y_bounds = [np.mean(det_y)] * 4
            phase_bounds, alpha = Trace.apply_polynomial_distortion(det_x_bounds, det_y_bounds, ai, bi)
            d = Globals.rule_spacing * Globals.ge_refracive_index
            wave_bounds = phase_bounds * 2.0 * d / ech_order
            wb_list.append(wave_bounds)
            if debug:
                det_x_rtn, det_y_rtn = Trace.apply_polynomial_distortion(phase_bounds, alpha, a, b)
                all_det_x_bounds.append(det_x_bounds)
                all_det_x_rtn.append(det_x_rtn)
                all_det_y_bounds.append(det_y_bounds)
                all_det_y_rtn.append(det_y_rtn)
            tfw = config, matrices, offset_correction, rays, wave_bounds
            tfw_list.append(tfw)
        if debug:
            title = "All Det X round trips "
            Plot.round_trip(np.array(all_det_x_bounds), np.array(all_det_x_rtn), title=title)
            title = "All Det Y round trips "
            Plot.round_trip(np.array(all_det_y_bounds), np.array(all_det_y_rtn), title=title)

        self.slice_objects = tfw_list
        return

    @staticmethod
    def get_fit_statistics(x, y, det_x, det_y):
        """ Calculate the mean and rms displacements between the pairs of coordinates, scaled by 1000
        to give output units in microns.
        """
        dis_x_list = []
        dis_y_list = []
        dis_x = x - det_x
        dis_y = y - det_y
        dis_x_list.append(dis_x)
        dis_y_list.append(dis_y)
        dx = np.asarray(dis_x_list)
        dy = np.asarray(dis_y_list)
        micron_mm = 1000.
        ave_x = micron_mm * np.mean(dx)
        ave_y = micron_mm * np.mean(dy)
        rms_x = micron_mm * np.std(dx)
        rms_y = micron_mm * np.std(dy)
        return ave_x, rms_x, ave_y, rms_y

    def plot_fit_maps(self, **kwargs):
        """ Plot ray coordinates at detector for the reference zemax data and also
        as projected using the passed list of transforms (one per slice). """
        suppress = kwargs.get('suppress', False)
        if suppress:
            return
        plot = Plot()

        plotdiffs = kwargs.get('plotdiffs', False)
        name = self.parameter['name']
        echelle_angle = self.parameter['Echelle angle']
        fig_title = "{:s} ea = {:4.2f}".format(name, echelle_angle)
        # SPIFU 1 column per spatial slice, 1 column
        n_rows, n_cols = 7, 4       # Nominal 1 slice per pane.
        unique_spifus, unique_slices = self.unique_spifu_slices, self.unique_slices
        spifu_start, slice_start = unique_spifus[0], unique_slices[0]
        if self.is_spifu:
            n_rows, = unique_spifus.shape
            n_cols, = unique_slices.shape
        xlim = (None if plotdiffs else [-60.0, 60.0])
        ax_list = plot.set_plot_area(fig_title,
                                     fontsize=10.0, sharex=True, sharey=False,
                                     nrows=n_rows, ncols=n_cols, xlim=xlim)
        for tf in self.slice_objects:
            config, _, _, rays, wave_coverage = tf
            ech_order, slice_no, spifu_no = config
            _, _, _, x, y, x_fit, y_fit = rays
            spifu_idx = int(spifu_no - spifu_start)
            slice_idx = int(slice_no - slice_start)
            if self.is_spifu:
                row = spifu_idx
                col = slice_idx
            else:
                row = int(slice_idx / n_cols)
                col = slice_idx % n_cols
            ax = ax_list[row, col]
            if plotdiffs:
                u, v = x - x_fit, y - y_fit
                ax.quiver(x, y, u, v, angles='xy')
            else:
                plot.plot_points(ax, x_fit, y_fit, ms=1.0, colour='blue')
                plot.plot_points(ax, x, y, ms=1.0, mk='x')
            fmt = "slice {:3.0f}, Sp slice {:3.0f}"
            ax.set_title(fmt.format(slice_no, spifu_no))
        plot.show()
        return

    def plot_scatter(self, tf, **kwargs):
        """ Plot the spread of offsets between ray trace and fit positions on the detector.
        """
        slice_list = kwargs.get('slice_list', None)
        plot_correction = kwargs.get('plot_correction', False)
        tlin1_default = "Distortion offsets (Zemax - Zemax Fit): \n"
        tlin1 = kwargs.get('tlin1', tlin1_default)

        if len(tf) == 5:
            config, _, offset_corrections, rays, wave_coverage = tf
        else:
            config, _, offset_corrections, rays = tf
        ech_order, slice_no, spifu_no = config
        off_x_corr, off_y_corr = offset_corrections
        # Create plot area
        ech_angle = self.parameter['Echelle angle']
        prism_angle = self.parameter['Prism angle']
        fmt = "order={:d}, ech. angle={:3.1f}, prism angle={:4.3f}, slice={:d}"
        tlin2 = fmt.format(int(ech_order), ech_angle, prism_angle, int(slice_no))
        if spifu_no >= 0.:
            tlin2 += ", spifu={:d}".format(int(spifu_no))
        title = tlin1 + tlin2

        ax_list = Plot.set_plot_area(title, nrows=1, ncols=2, fontsize=10.0, sharey=True)
        x_list, y_list = [], []
        off_det_x_list, off_det_y_list, off_det_a_list = [], [], []

        if slice_list is None or slice_no in slice_list:
            _, _, _, det_x, det_y, det_x_fit, det_y_fit = rays
            x_list.append(det_x)
            y_list.append(det_y)
            off_det_x, off_det_y = det_x - det_x_fit, det_y - det_y_fit
            off_det_a = np.sqrt(np.square(off_det_x) + np.square(off_det_y))
            off_det_x_list.append(off_det_x)
            off_det_y_list.append(off_det_y)
            off_det_a_list.append(off_det_a)

            for pane in range(0, 2):
                ax = ax_list[0, pane]
                u = det_x if pane == 0 else det_y
                if plot_correction and pane == 0:
                    xf = Trace.offset_error_function(off_x_corr, det_x, 0.)
                    yf = Trace.offset_error_function(off_y_corr, det_x, 0.)
                    uf, (vf,) = Trace._sort(u, (xf,))
                    ax.plot(uf, 1000.*vf, ls='solid', lw=1.5, color='salmon')
                    uf, (vf,) = Trace._sort(u, (yf,))
                    ax.plot(uf, 1000.*vf, ls='solid', lw=1.5, color='olive')
                ax.plot(u, 1000. * off_det_x, marker='.', ls='none', ms=1.5, color='orangered')
                ax.plot(u, 1000. * off_det_y, marker='.', ls='none', ms=1.5, color='green')
                ax.plot(u, 1000. * off_det_a, marker='.', ls='none', ms=1.5, color='blue')

        # Add the legend to both plots
        x_labels = ['Det x / mm', 'Det y / mm']
        y_labels = ["Offset / micron.", '']
        fmt = "{:s} offset stdev= {:3.1f} um"
        for pane in range(0, 2):
            ax = ax_list[0, pane]
            ax.set_xlabel(x_labels[pane])
            ax.set_ylabel(y_labels[pane])
            off_x_all = np.array(off_det_x_list) * 1000.
            off_y_all = np.array(off_det_y_list) * 1000.
            off_a_all = np.array(off_det_a_list) * 1000.
            off_x_rms = np.std(off_x_all)
            off_y_rms = np.std(off_y_all)
            off_a_rms = np.std(off_a_all)
            label1 = fmt.format('x', off_x_rms)
            label2 = fmt.format('y', off_y_rms)
            label3 = fmt.format('a', off_a_rms)
            patch_x = mpatches.Patch(color='salmon', label=label1)
            patch_y = mpatches.Patch(color='olive', label=label2)
            patch_a = mpatches.Patch(color='blue', label=label3)
            ax.legend(handles=[patch_x, patch_y, patch_a])

        Plot.show()
        return

    @staticmethod
    def _find_residual(det_xy, waves, poly_order):
        poly = np.polyfit(det_xy, waves, poly_order)
        text = poly.__str__()
        print(text)
        fit = np.poly1d(poly)
        wfit = fit(det_xy)
        dw = waves - wfit
        return dw

    @staticmethod
    def plot_dispersion(x, y, dw_dx, **kwargs):
        """ Plot distortion variation with detector coordinate v wavelength per echelle order
        """
        title = kwargs.get('title', 'Dispersion')
        ax_list = Plot.set_plot_area(title,
                                     nrows=1, ncols=1, fontsize=10.0)
        x_label = 'x(det), y(det) / mm'
        y_label = "$\delta\lambda/\delta$x / (nm / mm)"
        fmt = "{:s}(det)"
        label1, label2 = fmt.format('x'), fmt.format('y')

        ax = ax_list[0, 0]
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.plot(x, dw_dx, fillstyle='full', ms=10.0, marker='o', linestyle='None', color='blue')
        ax.plot(y, dw_dx, fillstyle='full', ms=10.0, marker='D', linestyle='None', color='green')
        patch_x = mpatches.Patch(color='blue', label=label1)
        patch_y = mpatches.Patch(color='green', label=label2)
        ax.legend(handles=[patch_x, patch_y])
        Plot.show()
        return

    def plot_slice_map(self, sno, **kwargs):
        """ Plot the planes in FP2 (nxlambda/2d, y) and at the detector (x, y)
        which are used to find the transforms """
        suppress = kwargs.get('suppress', False)
        if suppress:
            return

        plot = Plot()
        titles = ['LMS EFP', 'Detector']

        phase = self.get('phase', slice=sno)
        alpha = self.get('fp2_x', slice=sno)
        det_x = self.get('det_x', slice=sno)
        det_y = self.get('det_y', slice=sno)

        xlabels = ['Phase (n lambda / 2 d)', 'X [mm]']
        ylabels = ['alpha [mm]', 'Y [mm]']
        fig_title = "{:s}, slice= {:d}".format(self.parameter['name'], sno)
        ax_list = plot.set_plot_area(fig_title, nrows=1, ncols=2)

        xs = [phase, det_x]
        ys = [alpha, det_y]

        for idx in range(0, 2):
            x = xs[idx]
            y = ys[idx]
            xlim = self._find_limits(x, 0.1)
            ylim = self._find_limits(y, 0.1)
            ax = ax_list[0, idx]
            ax.set_title(titles[idx], loc='left')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xlabel(xlabels[idx])
            ax.set_ylabel(ylabels[idx])
            plot.plot_points(ax, x, y, fs='full', ms=1.0, mk='o', rgb=self.rgb)

        plot.show()
        return

    def plot_focal_planes(self, **kwargs):
        """ Plot coordinates at all focal surfaces. FP2_X v FP2_Y, FP4_X v Y

        :param kwargs:
        :return:
        """
        suppress = kwargs.get('suppress', False)
        if suppress:
            return

        focal_planes = Trace.spifu_focal_planes if self.is_spifu else Trace.nominal_focal_planes
        plot = Plot()
        n_focal_planes = len(focal_planes)

        xlabel = 'X [mm]'
        ylabel = 'Y [mm]'
        ax_list = plot.set_plot_area(self.__str__(), nrows=1, ncols=n_focal_planes)
        for pane, title in enumerate(focal_planes):
            ax = ax_list[0, pane]
            fp_x, fp_y = focal_planes[title]
            x = self.series[fp_x]
            y = self.series[fp_y]
            rgb = self.rgb
            ax.set_title(title, loc='left')
            ax.set_xlabel(xlabel)
            if pane == 0:
                ax.set_ylabel(ylabel)
            Plot.plot_points(ax, x, y, fs='full', ms=1.0, mk='o', rgb=rgb)
        plot.show()
        return

    @staticmethod
    def apply_polynomial_distortion(x, y, a, b):
        """
        @author Alistair Glasse
        21/2/17   Create to encapsulate distortion transforms
        Apply a polynomial transform pair (A,B or AI,BI) to an array of points
        affine = True  Apply an affine version of the transforms (remove all non-linear terms)
        """
        dim = a.shape[0]
        n_pts = len(x)

        exponent = np.array([range(0, dim), ] * n_pts).transpose()

        xmat = np.array([x, ] * dim)
        xin = np.power(xmat, exponent)
        ymat = np.array([y, ] * dim)
        yin = np.power(ymat, exponent)

        xout = np.zeros(n_pts)
        yout = np.zeros(n_pts)
        for i in range(0, n_pts):
            xout[i] = yin[:, i] @ a @ xin[:, i]
            yout[i] = yin[:, i] @ b @ xin[:, i]
        return xout, yout

    @staticmethod
    def apply_sine_correction(sine_corr, det_x, y_in, fwd):
        """ Apply sinusoidal correction.
        fwd = True for det_
        """
        y_corr = Trace.offset_error_function(sine_corr, det_x, 0.)
        y = y_in + y_corr
        return np.array(y)
