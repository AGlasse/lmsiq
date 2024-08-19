#!/usr/bin/env python
"""
@author Alistair Glasse
Python object to encapsulate ray trace data for the LMS

18/12/18  Created class (Glasse)
"""
import math
import numpy as np
import matplotlib.patches as mpatches
from lms_detector import Detector
from lms_globals import Globals
from lmsdist_util import Util
from lms_transform import Transform
from lmsdist_plot import Plot


class Trace:

    common_series_names = {'ech_order': 'int', 'slice': 'int', 'sp_slice': 'int',
                           'slicer_x': 'float', 'slicer_y': 'float', 'ifu_x': 'float', 'ifu_y': 'float'}
    nominal_efp_map = {'efp_x': 'fp2_x', 'efp_y': 'fp2_y'}
    nominal_series_names = {'slit_x': 'float', 'slit_y': 'float'}
    spifu_efp_map = {'efp_x': 'fp1_x', 'efp_y': 'fp1_y'}
    spifu_series_names = {'sp_slicer_x': 'float', 'sp_slicer_y': 'float'}

    nominal_focal_planes = {'LMS EFP': ('efp_x', 'efp_y'), 'Slicer': ('slicer_x', 'slicer_y'),
                            'IFU': ('ifu_x', 'ifu_y'), 'Slit': ('slit_x', 'slit_y'),
                            'Detector': ('det_x', 'det_y')}
    spifu_focal_planes = {'LMS EFP': ('efp_x', 'efp_y'), 'Slicer': ('slicer_x', 'slicer_y'),
                          'IFU': ('ifu_x', 'ifu_y'), 'SP slicer': ('sp_slicer_x', 'sp_slicer_y'),
                          'Detector': ('det_x', 'det_y')}

    def __init__(self, path, coord_in, coord_out, **kwargs):
        """ Read in Zemax ray trace data for a specific LMS configuration (echelle and prism angle, spectral IFU
        selected or not).
        @author: Alistair Glasse
        Read Zemax ray trace data
        """
        self.transforms = None                              # Replacement for slice_objects
        self.slice_objects, self.ray_trace = None, None
        self.offset_data = None
        self.n_mat_terms = None
        self.coord_in = coord_in
        self.coord_out = coord_out
        self.opticon = kwargs.get('opticon', False)
        is_spifu = self.opticon == Globals.spifu
        # fp_dict = Trace.spifu_focal_planes if self.is_spifu else Trace.nominal_focal_planes
        silent = kwargs.get('silent', False)
        if not silent:
            print('Reading Zemax model data from ' + path)
        efp_map = Trace.spifu_efp_map if is_spifu else Trace.nominal_efp_map
        csv_name, config, series = self._read_csv(path, is_spifu, coord_in, coord_out, efp_map)
        self.lms_config = {'opticon': self.opticon, 'pri_ang': config['Prism angle'],
                           'ech_ang': config['Echelle angle'], 'ech_order': config['Spectral order']}
        self.parameter = config

        # For SPIFU data, spifu slices 1,2 -> echelle order 23, 3,4 _> 24, 5,6 -> 25
        if is_spifu:
            spifus = series['sp_slice']
            spifu_indices = spifus - 1
            ech_orders = np.mod(spifu_indices, 3) + 23
            series['ech_order'] = ech_orders

        waves = series['wavelength']
        self.wmin, self.wmax = np.amin(waves), np.amax(waves)
        ech_orders = series['ech_order']
        self.series = series

        # Count the number of slices and rays
        self.unique_ech_orders = np.unique(ech_orders)
        slices = series['slice']
        self.unique_slices = np.unique(slices)
        spifus = series['sp_slice']
        self.unique_spifu_slices = np.unique(spifus)
        self.unique_waves = np.unique(waves)
        self.n_rays, = slices.shape

        self._create_mask(silent)
        return

    def __str__(self):
        fmt = "{:s}, EA={:6.3f} deg, PA={:6.3f} deg, "
        string = fmt.format(self.opticon, self.parameter['Echelle angle'], self.parameter['Prism angle'])
        smin, smax = self.unique_slices[0], self.unique_slices[-1]
        string += "slices {:d}-{:d}".format(int(smin), int(smax))
        return string

    def create_transforms(self, **kwargs):
        """ Generate transforms to map from phase, across-slice coordinates in the entrance focal plane, to detector
        row, column coordinates.  Also update the wavelength values at the centre and corners of the detector mosaic.
        """
        n_terms = Globals.transform_config['mat_order']
        debug = kwargs.get('debug', False)
        self.slice_objects = []
        off_x2_fits, off_y2_fits = [], []
        offset_data = []
        slice_order = n_terms - 1
        fp_in, fp_out = self.coord_in, self.coord_out
        is_first = True
        transforms = []
        for ech_order in self.unique_ech_orders:
            for spifu_no in self.unique_spifu_slices:
                for slice_no in self.unique_slices:
                    transform = Transform()

                    waves = self.get('wavelength', slice_no=slice_no, spifu_no=spifu_no)
                    ech_orders = self.get('ech_order', slice_no=slice_no, spifu_no=spifu_no)
                    alpha = self.get('efp_x', slice_no=slice_no, spifu_no=spifu_no)
                    det_x = self.get(fp_out[0], slice_no=slice_no, spifu_no=spifu_no)
                    det_y = self.get(fp_out[1], slice_no=slice_no, spifu_no=spifu_no)

                    phase = Util.waves_to_phases(waves, ech_orders)
                    a, b = Util.solve_svd_distortion(phase, alpha, det_x, det_y, slice_order, inverse=False)
                    ai, bi = Util.solve_svd_distortion(phase, alpha, det_x, det_y, slice_order, inverse=True)

                    det_x_fit1, det_y_fit1 = Util.apply_svd_distortion(phase, alpha, a, b)
                    off_x1, off_y1 = det_x - det_x_fit1, det_y - det_y_fit1
                    x_residual_correction = Util.fit_residual(det_x_fit1, det_x_fit1, off_x1)
                    y_residual_correction = Util.fit_residual(det_x_fit1, det_x_fit1, off_y1)

                    x_corr = Util.offset_error_function(x_residual_correction, det_x_fit1, 0.)
                    det_x_fit2 = det_x_fit1 + x_corr
                    y_corr = Util.offset_error_function(y_residual_correction, det_y_fit1, 0.)
                    det_y_fit2 = det_y_fit1 + y_corr

                    off_x2, off_y2 = det_x - det_x_fit2, det_y - det_y_fit2
                    off_x2_fits.append(off_x2)
                    off_y2_fits.append(off_y2)

                    tr_pars = self.lms_config.copy()
                    tr_pars.update(Globals.transform_config)
                    w_min, w_max = np.amin(waves), np.amax(waves)

                    config = ech_order, slice_no, spifu_no, w_min, w_max
                    matrices = a, b, ai, bi
                    offset_corrections = x_residual_correction, y_residual_correction
                    rays1 = waves, phase, alpha, det_x, det_y, det_x_fit1, det_y_fit1
                    tf_inter = config, matrices, offset_corrections, rays1
                    rays2 = waves, phase, alpha, det_x, det_y, det_x_fit2, det_y_fit2
                    slice_object = config, matrices, offset_corrections, rays2
                    self.slice_objects.append(slice_object)
                    if debug and is_first:        # Plot intermediate and full fit to data
                        tlin1 = "Distortion residuals, A,B, polynomial fit \n"
                        self.plot_scatter(tf_inter, plot_correction=True, tlin1=tlin1)
                        tlin1 = "Distortion residuals, post-sinusoidal correction \n"
                        self.plot_scatter(slice_object, tlin1=tlin1)
                        is_first = False
                    transforms.append(transform)
        self.transforms = transforms
        self.offset_data = offset_data
        return

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
            if self.opticon == Globals.spifu:
                spifu_slices = self.series['sp_slice']
                idx2 = spifu_slices == spifu_no
                idx = np.logical_and(idx, idx2)
            a = np.compress(idx, a)
        return a

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
        if self.opticon == Globals.spifu:
            n_rows, = unique_spifus.shape
            n_cols, = unique_slices.shape
        xlim = (None if plotdiffs else [-60.0, 60.0])
        ax_list = plot.set_plot_area(fig_title,
                                     fontsize=10.0, sharex=True, sharey=False,
                                     nrows=n_rows, ncols=n_cols, xlim=xlim)
        for tf in self.slice_objects:
            config, _, _, rays = tf
            ech_order, slice_no, spifu_no, _, _ = config
            _, _, _, x, y, x_fit, y_fit = rays
            spifu_idx = int(spifu_no - spifu_start)
            slice_idx = int(slice_no - slice_start)
            if self.opticon == Globals.spifu:
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
        tlin1_default = "Distortion residuals (Zemax - Zemax Fit): \n"
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
                    xf = Util.offset_error_function(off_x_corr, det_x, 0.)
                    yf = Util.offset_error_function(off_y_corr, det_x, 0.)
                    uf, (vf,) = Util.sort(u, (xf,))
                    ax.plot(uf, 1000.*vf, ls='solid', lw=1.5, color='salmon')
                    uf, (vf,) = Util.sort(u, (yf,))
                    ax.plot(uf, 1000.*vf, ls='solid', lw=1.5, color='olive')
                ax.plot(u, 1000. * off_det_x, marker='.', ls='none', ms=1.5, color='orangered')
                ax.plot(u, 1000. * off_det_y, marker='.', ls='none', ms=1.5, color='green')
                ax.plot(u, 1000. * off_det_a, marker='.', ls='none', ms=1.5, color='blue')

        # Add the legend to both plots
        x_labels = ['Det x / mm', 'Det y / mm']
        y_labels = ["Residual / micron.", '']
        fmt = "{:s} residual stdev= {:3.1f} um"
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

        :param kwargs:  suppress, True = suppress plot
                        colour_scheme, colour code points by colour_scheme = string
                                        'wavelength'
                                        'slice_no'
        :return:
        """
        suppress = kwargs.get('suppress', False)
        if suppress:
            return

        fp_plot_mask = {'LMS EFP': False, 'Slicer': False,
                        'IFU': False, 'Slit': True, 'SP slicer': True, 'Detector': True}

        is_spifu = self.opticon == Globals.spifu
        focal_planes = Trace.spifu_focal_planes if is_spifu else Trace.nominal_focal_planes

        n_focal_planes = len(focal_planes)
        colour_scheme = kwargs.get('colour_scheme', 'wavelength')
        rgb = self._create_wave_colours(colour_scheme)
        rgb_masked = self._apply_mask(rgb.copy())

        plot = Plot()

        xlabel = 'X [mm]'
        ylabel = 'Y [mm]'
        ax_list = plot.set_plot_area(self.__str__(), nrows=1, ncols=n_focal_planes)
        for pane, title in enumerate(focal_planes):
            if fp_plot_mask[title]:
                rgb = rgb_masked

            ax = ax_list[0, pane]
            fp_x, fp_y = focal_planes[title]
            x = self.series[fp_x]
            y = self.series[fp_y]
            ax.set_title(title, loc='left')
            ax.set_xlabel(xlabel)
            if pane == 0:
                ax.set_ylabel(ylabel)
            Plot.plot_points(ax, x, y, fs='full', ms=1.0, mk='o', rgb=rgb)
        plot.show()
        return

    def _read_csv(self, path, is_spifu, coord_in, coord_out, efp_map):
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
        efp_x_name = efp_map['efp_x']
        efp_y_name = efp_map['efp_y']
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
        zem_column = {}         # Dictionary to map zemax column names to column index and series name

        for i, token in enumerate(tokens):
            zem_tag = token.strip()
            ser_tag = zem_tag
            if ser_tag == efp_x_name:
                ser_tag = 'efp_x'
            if ser_tag == efp_y_name:
                ser_tag = 'efp_y'
            zem_column[ser_tag] = i, zem_tag

        series_names = Trace.common_series_names
        if is_spifu:
            series_names.update(Trace.spifu_series_names)
        else:
            series_names.update(Trace.nominal_series_names)

        fp_list = coord_in + coord_out
        for key in fp_list:
            series_names[key] = 'float'
        series = {}
        for name in series_names:
            series[name] = []
        while True:
            line = next(line_iter)
            tokens = line.split(',')
            if len(tokens) < 2:
                break
            for name in series_names:
                if name == 'ech_order':         # Assign spectral IFU orders later
                    val = 0 if is_spifu else parameter['Spectral order']
                    series[name].append(val)
                    continue
                if name == 'spifu_slice':
                    val = 0
                    if is_spifu:
                        zem_col, zem_tag = zem_column[name]
                        val = int(tokens[zem_col])
                    series[name].append(val)      # Default for nominal configuration
                    continue
                zem_col, zem_name = zem_column[name]
                fmt = series_names[name]
                val = None
                token = tokens[zem_col]
                match fmt:
                    case 'float':
                        val = float(token)
                    case 'int':
                        val = int(token)
                    case _:
                        print("Un-supported format {:s} for token {:s}".format(fmt, token))
                series[name].append(val)
        # Convert series lists to numpy arrays
        int_arrays = ['ech_order', 'slice', 'spifu_slice']
        for name in series:
            vals = series[name]
            series[name] = np.array(vals, dtype=int) if name in int_arrays else np.array(vals)
        return csv_name, parameter, series

    def _create_mask(self, silent):
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

    def _apply_mask(self, rgb):
        mask = self.mask
        mask_indices = np.argwhere(mask == 0)
        grey = [.3, .3, .3]
        rgb[mask_indices, :] = grey
        return rgb

    def _create_wave_colours(self, colour_scheme):

        slices = self.series['slice']
        rgb = None
        if colour_scheme == 'wavelength':
            waves = self.series['wavelength']
            rgb = Util.make_rgb_gradient(waves)
        return rgb

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
