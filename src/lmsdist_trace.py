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
from lmsdist_plot import Plot
from lms_transform import Transform


class Trace:

    common_series_names = {'ech_order': 'int', 'slice_no': 'int', 'sp_slice': 'int',
                           'slicer_x': 'float', 'slicer_y': 'float',
                           'ifu_x': 'float', 'ifu_y': 'float'}
    nominal_efp_map = {'efp_x': 'fp2_x', 'efp_y': 'fp2_y'}
    nominal_series_names = {'slit_x': 'float', 'slit_y': 'float'}
    spifu_efp_map = {'efp_x': 'fp1_x', 'efp_y': 'fp1_y'}
    spifu_series_names = {'sp_slicer_x': 'float', 'sp_slicer_y': 'float'}

    nominal_focal_planes = {'LMS EFP': ('efp_x', 'efp_y'),
                            'Slicer': ('slicer_x', 'slicer_y'),
                            'IFU': ('ifu_x', 'ifu_y'),
                            'Slit': ('slit_x', 'slit_y'),
                            'Detector': ('det_x', 'det_y')}
    spifu_focal_planes = {'LMS EFP': ('efp_x', 'efp_y'),
                          'Slicer': ('slicer_x', 'slicer_y'),
                          'IFU': ('ifu_x', 'ifu_y'),
                          'SP slicer': ('sp_slicer_x', 'sp_slicer_y'),
                          'Detector': ('det_x', 'det_y')}
    cfg_tags, cfg_id_counter = [], 0

    affines, inverse_affines = None, None      # Global MFP <-> DFP transforms.  Written once during __init__

    def __init__(self, path, model_config, **kwargs):
        """ Read in Zemax ray trace data for a specific LMS configuration (echelle and prism angle, spectral IFU
        selected or not).
        @author: Alistair Glasse
        Read Zemax ray trace data
        """
        _, opticon, date_stamp, optical_path_label, coord_in, coord_out = model_config
        self.is_spifu = opticon == Globals.extended
        self.model_config = model_config
        self.transform_fits_name = None
        self.ray_trace = None
        self.transforms = []
        self.offset_data = None
        self.a_rms = 0.                          # RMS transform errorfor trace (microns)
        self.n_mat_terms = None
        silent = kwargs.get('silent', False)
        if not silent:
            print('Reading Zemax model data from ' + path)
        csv_name, config, series = self._read_csv(path, model_config)
        pri_ang = config['Prism angle']
        ech_ang = config['Echelle angle']
        # Set unique (LMS mechanism settings) configuration number for this trace.
        cfg_tag = "{:s}_{:05.3f}_{:05.3f}".format(opticon, ech_ang, pri_ang)
        if cfg_tag not in Trace.cfg_tags:
            self.cfg_id = Trace.cfg_id_counter
            Trace.cfg_id_counter += 1
        self.lms_config = {'opticon': opticon, 'cfg_id': self.cfg_id, 'pri_ang': config['Prism angle'],
                           'ech_ang': config['Echelle angle'], 'ech_order': config['Spectral order']}
        self.parameter = config

        # For SPIFU data, spifu slices 1,2 -> echelle order 23, 3,4 _> 24, 5,6 -> 25
        if self.is_spifu:
            spifus = series['sp_slice']
            spifu_indices = spifus - 1
            ech_orders = spifu_indices // 2 + 23
            series['ech_order'] = ech_orders

        waves = series['wavelength']
        self.wmin, self.wmax = np.amin(waves), np.amax(waves)
        self.series = series

        # Count the number of slices and rays
        ech_orders = series['ech_order']
        self.unique_ech_orders = np.unique(ech_orders)
        slice_nos = series['slice_no']
        self.unique_slices = np.unique(slice_nos)
        spifus = series['sp_slice']
        self.unique_spifu_slices = np.unique(spifus)
        self.unique_waves = np.unique(waves)
        self.n_rays, = slice_nos.shape

        self._create_mask(silent)
        Trace.model_configuration = model_config
        Trace.create_affine_transforms()
        return

    def __str__(self):
        fmt = "{:s}, EA={:6.3f} deg, PA={:6.3f} deg, "
        opticon = self.model_config[1]
        string = fmt.format(opticon, self.parameter['Echelle angle'], self.parameter['Prism angle'])
        smin, smax = self.unique_slices[0], self.unique_slices[-1]
        string += "slices {:d}-{:d}".format(int(smin), int(smax))
        return string

    @staticmethod
    def create_affine_transforms():
        """ Create the four (global) matrices which map points in the mosaic focal plane (in units of mm)
        to pixel row and column positions.
        """
        n_dets = 4
        aff_shape = 2*n_dets, 3, 3
        affines = np.zeros(aff_shape) * n_dets
        thetas = [0.]*4
        xy_fc = Globals.det_size + Globals.det_gap / 2.
        xy_nc = Globals.det_gap / 2.
        x_mfp_origins = [-xy_fc, +xy_nc, -xy_fc, +xy_nc]
        y_mfp_origins = [-xy_nc, -xy_nc, +xy_fc, +xy_fc]
        pix_mm = 1000. / Globals.nom_pix_pitch
        y_scales = [pix_mm] * 4
        x_scales = [-pix_mm] * 4
        for i in range(0, n_dets):
            sx, sy, theta, x_mfp_org, y_mfp_org = x_scales[i], y_scales[i], thetas[i], x_mfp_origins[i], y_mfp_origins[i]
            cos_theta, sin_theta = math.cos(theta), math.sin(theta)
            affines[i, 0, :] = [sx * cos_theta, -sy * sin_theta, sx * x_mfp_org]
            affines[i, 1, :] = [sx * sin_theta, +sy * cos_theta, sy * y_mfp_org]
            affines[i, 2, :] = [0., 0., 1.]
            affines[i+n_dets] = np.linalg.inv(affines[i])
        Trace.affines = affines
        return

    def get_ifp_boresight(self, opticon):
        spifu_no = 0 if opticon == Globals.nominal else 3
        ech_order = self.unique_ech_orders[0] if opticon == Globals.nominal else self.unique_ech_orders[1]
        slit_x = self.get('ifu_x', slice_no=13, spifu_no=spifu_no, ech_order=ech_order)
        waves = self.get('wavelength', slice_no=13, spifu_no=spifu_no, ech_order=ech_order)

        ech_orders = self.get('ech_order', slice_no=13, spifu_no=spifu_no, ech_order=ech_order)
        wave_bs = np.interp(0.0, slit_x, waves)     # Find wavelength where slit_x == 0.
        pri_ang = self.lms_config['pri_ang']
        return wave_bs, pri_ang

    def create_transforms(self, **kwargs):
        """ Generate transforms to map from phase, across-slice coordinates in the entrance focal plane, to detector
        row, column coordinates.  Also update the wavelength values at the centre and corners of the detector mosaic.
        """
        debug = kwargs.get('debug', False)
        n_terms = Globals.svd_order
        slice_order = n_terms - 1
        _, _, _, _, fp_in, fp_out = self.model_config
        is_first = True
        a_rms_list = []
        for spifu_no in self.unique_spifu_slices:
            for slice_no in self.unique_slices:
                for ech_order in self.unique_ech_orders:
                    transform = Transform(cfg=self.lms_config)
                    cfg = transform.configuration
                    cfg['slice_no'] = slice_no
                    cfg['spifu_no'] = spifu_no
                    cfg['ech_order'] = ech_order
                    # kwargs = {'spifu_no': spifu_no, 'slice_no': slice_no, 'ech_order': ech_order}
                    waves = self.get('wavelength', **cfg)
                    n_waves, = waves.shape
                    if n_waves == 0:
                        continue

                    ech_orders = self.get('ech_order', **cfg)
                    print(slice_no, spifu_no, ech_order, waves[0], ech_orders[0])

                    alpha = self.get('efp_x', **cfg)
                    mfp_x = self.get(fp_out[0], **cfg)
                    mfp_y = self.get(fp_out[1], **cfg)

                    phase = Util.waves_to_phases(waves, ech_orders)
                    a, b = Util.solve_svd_distortion(phase, alpha, mfp_x, mfp_y, slice_order, inverse=False)
                    ai, bi = Util.solve_svd_distortion(phase, alpha, mfp_x, mfp_y, slice_order, inverse=True)

                    mfp_x_fit, mfp_y_fit = Util.apply_svd_distortion(phase, alpha, a, b)
                    off_mfp_x, off_mfp_y = mfp_x - mfp_x_fit, mfp_y - mfp_y_fit
                    off_mfp_a = np.sqrt(np.square(off_mfp_x) + np.square(off_mfp_y))
                    a_rms_list.append(off_mfp_a)

                    w_min, w_max = np.amin(waves), np.amax(waves)
                    cfg['w_min'] = w_min
                    cfg['w_max'] = w_max
                    mats = transform.matrices
                    mats['a'] = a
                    mats['b'] = b
                    mats['ai'] = ai
                    mats['bi'] = bi
                    cfg['n_mats'] = 4
                    cfg['mat_order'], _ = a.shape
                    rays = waves, phase, alpha, mfp_x, mfp_y, mfp_x_fit, mfp_y_fit
                    if debug and is_first:        # Plot intermediate and full fit to data
                        fmt = "Distortion residuals, A,B, polynomial fit, SVD cutoff = {:5.1e}\n"
                        tlin1 = fmt.format(Globals.svd_cutoff)
                        self.plot_scatter(transform, rays, plot_correction=True, tlin1=tlin1)
                        is_first = False
                    self.transforms.append(transform)

        a_rms = np.sqrt(np.mean(np.square(np.array(a_rms_list))))
        self.a_rms = a_rms
        return

    def get(self, tag, **kwargs):
        """ Extract a specific coordinate (identified by 'tag') from the trace for rays which pass through
        a specified spatial and (if the spectral IFU is selected) spectral slice.
        """
        _, opticon, _, _, _, _ = self.model_config
        slice_no = kwargs.get('slice_no', None)
        spifu_no = kwargs.get('spifu_no', None)
        ech_order = kwargs.get('ech_order', None)
        a = self.series[tag]
        if slice_no is not None:
            slice_nos = self.series['slice_no']
            idx = slice_nos == slice_no
            if opticon == Globals.extended:
                spifu_slices = self.series['sp_slice']
                idx2 = spifu_slices == spifu_no
                idx = np.logical_and(idx, idx2)
                ech_orders = self.series['ech_order']
                idx3 = ech_orders == ech_order
                idx = np.logical_and(idx, idx3)
            a = np.compress(idx, a)
        return a

    def plot_fit_maps(self, **kwargs):
        """ Plot ray coordinates at detector for the reference zemax data and also as projected using the
        passed list of transforms (one per slice).
        """
        suppress = kwargs.get('suppress', False)
        if suppress:
            return
        plot = Plot()

        plotdiffs = kwargs.get('plotdiffs', False)
        field = kwargs.get('field', False)
        subset = kwargs.get('subset', False)
        name = self.parameter['name']
        echelle_angle = self.parameter['Echelle angle']
        fig_title = "{:s} ea = {:4.2f}".format(name, echelle_angle)
        # SPIFU 1 column per spatial slice, 1 column
        n_rows, n_cols = 7, 4
        unique_spifus, unique_slices = self.unique_spifu_slices, self.unique_slices
        spifu_start, slice_start = unique_spifus[0], unique_slices[0]
        if self.is_spifu:
            n_rows, = unique_spifus.shape
            n_cols, = unique_slices.shape
        if subset:
            n_rows = 3
            n_cols = 1
        xlim = (None if plotdiffs else [-40.0, 40.0])
        fig, ax_list = plot.set_plot_area(fontsize=14.0, sharex=True, sharey=False,
                                     nrows=n_rows, ncols=n_cols, xlim=xlim)
        fig.suptitle(fig_title)
        for ifu_slice in self.slices:
            config, _, rays = ifu_slice
            ech_order, slice_no, spifu_no, _, _ = config
            row, col = -1, 0
            if subset:
                if self.is_spifu:
                    select_row = {6: 0, 3: 1, 1: 2}
                    if slice_no == 12 and spifu_no in select_row:
                        row = select_row[spifu_no]
                else:
                    select_row = {28: 0, 12: 1, 1: 2}
                if slice_no in select_row:
                    row = select_row[slice_no]
            else:
                spifu_idx = int(spifu_no - spifu_start)
                slice_idx = int(slice_no - slice_start)
                if self.is_spifu:
                    row = spifu_idx
                    col = slice_idx
                else:
                    row = int(slice_idx / n_cols)
                    col = slice_idx % n_cols

            if row < 0:       # Skip plot if the pane row is not valid.
                continue
            ax = ax_list[row, col]
            _, _, _, x, y, x_fit, y_fit = rays
            if field:                       # Draw filled polygon around field points.
                n_uw = len(self.unique_waves)
                n_pts = 2 * n_uw + 1        # No. of points on perimeter
                xp, yp = np.zeros(n_pts), np.zeros(n_pts)
                alphas = rays[2]
                uni_alphas = np.unique(alphas)
                idxs = np.argwhere(alphas == uni_alphas[0]).flatten()
                xp[0:n_uw], yp[0:n_uw] = x[idxs], y[idxs]
                idxs = np.argwhere(alphas == uni_alphas[-1]).flatten()
                xp[n_uw:-1], yp[n_uw:-1] = np.flip(x[idxs]), np.flip(y[idxs])
                xp[-1], yp[-1] = xp[0], yp[0]
                ax.fill(xp, yp, color='pink')
            if plotdiffs:
                u, v = x - x_fit, y - y_fit
                q = ax.quiver(x_fit, y_fit, u, v,
                              angles='xy', scale_units='xy', scale=.001,
                              width=0.001)
                if row == 0:
                    ax.quiverkey(q, X=0.9, Y=1.1, U=0.01, label='10 microns', labelpos='N')
            else:
                plot.plot_points(ax, x_fit, y_fit, ms=1.0, colour='blue')
                plot.plot_points(ax, x, y, ms=1.0, mk='x')
            title = "slice {:3.0f}".format(slice_no)
            if self.is_spifu:
                title += ", spectral slice {:3.0f}".format(spifu_no)
            ax.set_title(title)
        ax = ax_list[n_rows-1, 0]
        ax.set_xlabel('x$_{mfp}$ / mm')
        ax.set_ylabel('y$_{mfp}$ / mm')

        plot.show()
        return

    def plot_scatter(self, transform, rays, **kwargs):
        """ Plot the spread of offsets between ray trace and fit positions on the detector.
        """
        slice_list = kwargs.get('slice_list', None)
        tlin1_default = "Distortion residuals (Zemax - Zemax Fit): \n"
        tlin1 = kwargs.get('tlin1', tlin1_default)
        config = transform.configuration

        ech_order = config['ech_order']
        slice_no = config['slice_no']
        spifu_no = config['spifu_no']
        # Create plot area
        ech_angle = self.parameter['Echelle angle']
        prism_angle = self.parameter['Prism angle']
        fmt = "order={:d}, ech. angle={:3.1f}, prism angle={:4.3f}, slice={:d}"
        tlin2 = fmt.format(int(ech_order), ech_angle, prism_angle, int(slice_no))
        if spifu_no > 0:
            tlin2 += ", spifu={:d}".format(int(spifu_no))
        title = tlin1 + tlin2

        fig, ax_list = Plot.set_plot_area(nrows=1, ncols=2, fontsize=10.0, sharey=True)
        fig.suptitle(title)

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
                ax.plot(u, 1000. * off_det_x, marker='.', ls='none', ms=1.5, color='orangered')
                ax.plot(u, 1000. * off_det_y, marker='.', ls='none', ms=1.5, color='green')
                ax.plot(u, 1000. * off_det_a, marker='.', ls='none', ms=1.5, color='blue')

        # Add the legend to both plots
        x_labels = ['Det x / mm', 'Det y / mm']
        y_labels = ["Residual / micron.", '']
        fmt = "{:s} residual, $\sigma$ = {:4.2f} $\mu$m"
        for pane in range(0, 2):
            ax = ax_list[0, pane]
            ax.set_xlabel(x_labels[pane])
            ax.set_ylabel(y_labels[pane])
            if pane == 0:       # Only draw the key in pane 0
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
        focal_planes = Trace.spifu_focal_planes if self.is_spifu else Trace.nominal_focal_planes

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

    def _read_csv(self, path, model_config):
        """ Read trace data in from csv file pointed to by path
        :return:
        """
        _, _, _, _, coord_in, coord_out = model_config
        is_spifu = self.is_spifu
        csv_name = path.split('/')[-1]
        name = csv_name.split('.')[0]
        with open(path, 'r') as text_file:
            read_data = text_file.read()
        line_list = read_data.split('\n')
        line_iter = iter(line_list)

        # Read configuration parameters
        efp_map = Trace.spifu_efp_map if is_spifu else Trace.nominal_efp_map
        efp_x_name = efp_map['efp_x']
        efp_y_name = efp_map['efp_y']
        parameter = {'name': name}
        while True:
            line = next(line_iter)
            tokens = line.split(':')
            if len(tokens) < 2:
                break
            name = tokens[0].strip()
            val = float(tokens[1].strip(', '))
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
            if ser_tag == 'slice':
                ser_tag = 'slice_no'
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
                if name == 'sp_slice':
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
        edge = Detector.det_size * Detector.det_pix_size / 1000.0
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

        rgb = None
        if colour_scheme == 'wavelength':
            waves = self.series['wavelength']
            rgb = Plot.make_rgb_gradient(waves)
        return rgb

    @staticmethod
    def _find_limits(a, margin):
        amin = min(a)
        amax = max(a)
        arange = amax - amin
        amargin = margin * arange
        limits = [amin - amargin, amax + amargin]
        return limits
