#!/usr/bin/env python
"""
@author Alistair Glasse
Python object to encapsulate ray trace data for the LMS

18/12/18  Created class (Glasse)
"""
import math
import numpy as np
from scipy.optimize import least_squares
from lms_detector import Detector
from lms_globals import Globals
from lms_dist_util import Util
from lms_dist_plot import Plot
import matplotlib.patches as mpatches


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
    parameter, column, data, phase = None, None, None, None
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
        self.tf_list, self.ray_trace = None, None
        self.offset_data, self.wave_bounds = None, None
        self.coord_in = coord_in
        self.coord_out = coord_out
        is_spifu = kwargs.get('is_spifu', False)
        self.is_spifu = is_spifu
        silent = kwargs.get('silent', False)
        if not silent:
            print('Reading Zemax model data from ' + path)

        csv_name, parameter, series = self._read_csv(path, is_spifu, coord_in, coord_out)

        # For SPIFU data, spifu slices 1,2 -> echelle order 23, 3,4 _> 24, 5,6 -> 25
        if is_spifu:
            spifu_slices = series['spifu_slice']
            spifu_indices = spifu_slices - 1.
            ech_orders = np.mod(spifu_indices, 3) + 23.
            series['ech_order'] = ech_orders

        waves = series['wavelength']
        ech_orders = series['ech_order']
        phase = self._waves_to_phase(waves, ech_orders)
        d = Globals.rule_spacing * Globals.ge_refracive_index
        sin_aoi = waves * ech_orders / (2.0 * d)
        phase = np.arcsin(sin_aoi)
        f = 1 - waves / np.mean(waves)
        k1, k2 = 0.000006, 375.
        phase_corr = k1 * np.sin(k2 * f)
        series['phase'] = phase + phase_corr
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
#        self.plot_phase_correction(f, phase_corr)       # self.get('det_x')
        return

    @staticmethod
    def _waves_to_phase(waves, ech_orders):
        d = Globals.rule_spacing * Globals.ge_refracive_index
        phase = waves * ech_orders / (2.0 * d)
        return phase

    def __str__(self):
        fmt = "EA={:6.3f} deg, PA={:6.3f} deg, "
        string = fmt.format(self.parameter['Echelle angle'], self.parameter['Prism angle'])
        string += '+SpIFU, ' if self.is_spifu else ''
        smin, smax = self.unique_slices[0], self.unique_slices[-1]
        string += "slices {:d}-{:d}".format(int(smin), int(smax))
        return string

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
                        val = 0. if is_spifu else parameter['Spectral order']
                    else:
                        if name == 'spifu_slice':
                            val = -1.       # Default for nominal configuration
                            if is_spifu:
                                zem_col = zem_column['sp_slice']
                                val = float(tokens[zem_col])
                        else:
                            zem_col = zem_column[name]
                            val = float(tokens[zem_col])
                series[name].append(val)
        # Convert series lists to numpy arrays
        for name in series:
            values = series[name]
            series[name] = np.array(values)

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
    def offset_error_function(p, x, y):
        two_pi = 2. * math.pi
        residual = p[0] * np.sin(two_pi * x / p[1]) - y
        return residual

    @staticmethod
    def _sort_xy(x_in, y_in):
        indices = np.argsort(x_in)
        x = x_in[indices]
        y = y_in[indices]
        return x, y


    @staticmethod
    def _sine_fit(x_in, y_in):
        x, y = Trace._sort_xy(x_in, y_in)
        guess_amplitude = 0.5 * (np.max(y) - np.min(y))
        guess_period = (2./3.) * (np.max(x) - np.min(x))
        guess_oc = [guess_amplitude, guess_period]
        res_lsq = least_squares(Trace.offset_error_function, guess_oc, args=(x, y))
        offset_correction = res_lsq.x
        fmt = "p_in={:8.6f},{:5.1f} -> p_out={:8.6f},{:5.1f}"
        print(fmt.format(guess_oc[0], guess_oc[1], offset_correction[0], offset_correction[1]))
        return offset_correction

    def create_transforms(self, n_terms, **kwargs):
        """ Generate transforms to map from phase, across-slice coordinates in the entrance focal plane, to detector
        row, column coordinates.  Also update the wavelength values at the centre and corners of the detector mosaic.
        """
        debug = kwargs.get('debug', False)
        util = Util()
        self.tf_list = []
        off_x_fits = []
        offset_data = []
        tf_order = n_terms - 1
        fp_in, fp_out = self.coord_in, self.coord_out
        for ech_order in self.unique_ech_orders:
            for spifu_no in self.unique_spifu_slices:
                for slice_no in self.unique_slices:
                    waves = self.get('wavelength', slice_no=slice_no, spifu_no=spifu_no)
                    phase = self.get(fp_in[0], slice_no=slice_no, spifu_no=spifu_no)
                    alpha = self.get(fp_in[1], slice_no=slice_no, spifu_no=spifu_no)
                    det_x = self.get(fp_out[0], slice_no=slice_no, spifu_no=spifu_no)
                    det_y = self.get(fp_out[1], slice_no=slice_no, spifu_no=spifu_no)

                    a, b = util.distortion_fit(phase, alpha, det_x, det_y, tf_order, inverse=False)
                    ai, bi = util.distortion_fit(phase, alpha, det_x, det_y, tf_order, inverse=True)
                    det_x_fit, det_y_fit = Util.apply_distortion(phase, alpha, a, b)
                    off_x, off_y = det_x - det_x_fit, det_y - det_y_fit

                    offset_correction = self._sine_fit(det_x_fit, off_x)

                    off_x_fit = self.offset_error_function(offset_correction, det_x_fit, 0.*det_x_fit)
                    off_x_fits.append(off_x_fit)
                    offset_data.append([off_x, off_y])
                    if debug:
                        phase_rt, alpha_rt = Util.apply_distortion(det_x_fit, det_y_fit, ai, bi)
                        print('Trace.create_transforms debug=True')
                        fmt = "Phase round trip EO= {:d}, slice_no= {:d}, spifu_no= {:d}"
                        title = fmt.format(int(ech_order), int(slice_no), int(spifu_no))
                        Plot.round_trip(phase, phase_rt, title=title)
                    config = ech_order, slice_no, spifu_no
                    matrices = a, b, ai, bi
                    rays = waves, phase, alpha, det_x, det_y, det_x_fit, det_y_fit
                    tf = config, matrices, offset_correction, rays

                    self.tf_list.append(tf)
                    self.plot_scatter(xfit=1000.*np.array(off_x_fits))

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
        for tf in self.tf_list:
            config, matrices, offset_correction, rays = tf
            ech_order, slice_no, spifu_no = config
            a, b, ai, bi = matrices
            _, _, _, _, det_y, _, _ = rays
            det_y_bounds = [np.mean(det_y)] * 4
            phase_bounds, alpha = Util.apply_distortion(det_x_bounds, det_y_bounds, ai, bi)
            d = Globals.rule_spacing * Globals.ge_refracive_index
            wave_bounds = phase_bounds * 2.0 * d / ech_order
            wb_list.append(wave_bounds)
            if debug:
                det_x_rtn, det_y_rtn = Util.apply_distortion(phase_bounds, alpha, a, b)
                all_det_x_bounds.append(det_x_bounds)
                all_det_x_rtn.append(det_x_rtn)
                all_det_y_bounds.append(det_y_bounds)
                all_det_y_rtn.append(det_y_rtn)
            tfw = config, matrices, rays, wave_bounds
            tfw_list.append(tfw)
        if debug:
            title = "All Det X round trips "
            Plot.round_trip(np.array(all_det_x_bounds), np.array(all_det_x_rtn), title=title)
            title = "All Det Y round trips "
            Plot.round_trip(np.array(all_det_y_bounds), np.array(all_det_y_rtn), title=title)

        self.tf_list = tfw_list
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
        for tf in self.tf_list:
            config, _, rays, wave_coverage = tf
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

    def plot_scatter(self, **kwargs):
        """ Plot the spread of offsets between ray trace and fit positions on the detector.
        """
        slice_list = kwargs.get('slice_list', None)
        xfit = kwargs.get('xfit', None)

        ax_list = Plot.set_plot_area('Distortion offsets (Zemax - Zemax Fit)',
                                     nrows=1, ncols=2, fontsize=10.0, sharey=True)
        x_list, y_list = [], []
        off_x_list, off_y_list = [], []
        for tf in self.tf_list:
            if len(tf) == 5:
                config, _, _, rays, wave_coverage = tf
            else:
                config, _, _, rays = tf
            _, slice_no, spifu_no = config
            if slice_list is None or slice_no in slice_list:
                _, _, _, x, y, x_fit, y_fit = rays
                x_list.append(x)
                y_list.append(y)
                off_x, off_y = x - x_fit, y - y_fit
                off_x_list.append(off_x)
                off_y_list.append(off_y)
        x_all, y_all = np.array(x_list), np.array(y_list)
        off_x_all = np.array(off_x_list) * 1000.
        off_y_all = np.array(off_y_list) * 1000.
        off_x_rms = np.std(off_x_all)
        off_y_rms = np.std(off_y_all)

        x_labels = ['Det x / mm', 'Det y / mm']
        y_labels = ["Offset / micron.", '']
        fmt = "{:s} offset stdev= {:3.1f} um"
        label1, label2 = fmt.format('x', off_x_rms), fmt.format('y', off_y_rms)
        for pane in range(0, 2):
            ax = ax_list[0, pane]
            ax.set_xlabel(x_labels[pane])
            ax.set_ylabel(y_labels[pane])
            absc = x_all if pane == 0 else y_all
            if xfit is not None:
                ax.plot(absc, xfit, ls='solid', lw=1.5, color='black')
            ax.plot(absc, off_x_all, marker='.', ls='none', ms=1.0, color='blue')
            ax.plot(absc, off_y_all, marker='.', ls='none', ms=1.0, color='green')

            patch_x = mpatches.Patch(color='blue', label=label1)
            patch_y = mpatches.Patch(color='green', label=label2)
            ax.legend(handles=[patch_x, patch_y])
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

    def plot_phase_correction(self, det_x, phase_corr):
#        poly_order = kwargs.get('poly_order', 0)            # Linear fit is default (poly_order = 0)
        title = "Phase correction"
        ax_list = Plot.set_plot_area(title,
                                     nrows=1, ncols=1, fontsize=10.0)
        x_label = "det_x / mm"
        y_label = "phase correction"
        fmt = "{:s} offset"
        label1, label2 = fmt.format('x'), fmt.format('y')

        ax = ax_list[0, 0]
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        ax.plot(det_x, phase_corr, fillstyle='full', ms=1.0, marker='o', linestyle='None', color='blue')
        # patch_x = mpatches.Patch(color='blue', label=label1)
        # patch_y = mpatches.Patch(color='green', label=label2)
        # ax.legend(handles=[patch_x, patch_y])
        Plot.show()
        return

    def plot_dispersion(self, **kwargs):
        """ Plot detector coordinate v wavelength per echelle order
        """
        poly_order = kwargs.get('poly_order', 0)            # Linear fit is default (poly_order = 0)
        title = "Dispersion offset, poly order= {:d}".format(poly_order)
        ax_list = Plot.set_plot_area(title,
                                     nrows=1, ncols=2, fontsize=10.0, sharey=True)
        x_labels = ['Det x / mm', 'Det y / mm']
        y_labels = ["Wave. res. / micron.", ""]
        fmt = "{:s} offset"
        label1, label2 = fmt.format('x'), fmt.format('y')

        for pane in range(0, 2):
            ax = ax_list[0, pane]
            ax.set_xlabel(x_labels[pane])
            ax.set_ylabel(y_labels[pane])

            for slice_no in self.unique_slices[0:1]:
                waves = self.get('wavelength', slice_no=slice_no)
                det_x = self.get('det_x', slice_no=slice_no)
                det_y = self.get('det_y', slice_no=slice_no)
                dw_x = self._find_residual(det_x, waves, poly_order)
                dw_y = self._find_residual(det_y, waves, poly_order)

                absc = det_x if pane == 0 else det_y
                ax.plot(absc, dw_x, fillstyle='full', ms=1.0, marker='o', linestyle='None', color='blue')
                ax.plot(absc, dw_y, fillstyle='full', ms=1.0, marker='o', linestyle='None', color='green')
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

    def tfs_to_text(self, tf_list):
        from lms_dist_util import Util

        util = Util()
        n_terms, _ = tf_list[0][3].shape
        fp_in, fp_out = self.coord_in, self.coord_out
        text = ''
        for tf_idx, tf in enumerate(tf_list):
            eo, sno, spifu_no, a, b, ai, bi = tf
            phase = self.get(fp_in[0], slice=sno, spifu_slice=spifu_no)
            alpha = self.get(fp_in[1], slice=sno, spifu_slice=spifu_no)
            det_x = self.get(fp_out[0], slice=sno, spifu_slice=spifu_no)
            det_y = self.get(fp_out[1], slice=sno, spifu_slice=spifu_no)

            x, y = util.apply_distortion(phase, alpha, a, b)
            ave_x, rms_x, ave_y, rms_y = self.get_fit_statistics(x, y, det_x, det_y)
            ea, pa, w1, w2, w3, w4 = self._get_parameters()
            for i in range(0, 4):
                mat = tf[i + 3]
                for j in range(0, n_terms):

                    fmt = "{:3.0f},{:6.3f},{:6.3f},{:6.0f},{:6.0f},"
                    text += fmt.format(eo, ea, pa, sno, spifu_no)
                    fmt = "{:6d},{:6d},{:6d},"
                    text += fmt.format(tf_idx, i, j)
                    fmt = '{:15.7e},'
                    for k in range(0, n_terms):
                        text += fmt.format(mat[j, k])

                    fmt = "{:8.3f},{:8.3},{:8.3f},{:8.3f},"
                    text += fmt.format(w1, w2, w3, w4)
                    fmt = "{:8.1f},{:8.1f},{:8.1f},{:8.1f}\n"
                    text += fmt.format(ave_x, rms_x, ave_y, rms_y)
        return text
