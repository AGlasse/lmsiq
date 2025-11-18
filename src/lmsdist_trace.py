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
# from src.lmsdist import efp_points


class Trace:

    # Configuration dicts.  Note lms_config is instantiated from Globals.
    series_names = {'ech_ord': 'int', 'slice_no': 'int', 'sp_slice': 'int',
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
    model_config = None

    def __init__(self, **kwargs):
        self.lms_config = None
        self.transform_fits_name, self.csv_name = None, None
        self.transforms = []
        self.offset_data = None
        self.a_rms = 0.                          # RMS transform error for trace (microns)
        self.n_mat_terms = None
        self.is_extended = None
        self.series = None
        self.unique_ech_ords = None
        self.unique_slices = None
        self.unique_spifu_slices = None
        self.unique_waves = None
        self.n_rays = None
        return

    def load(self, path, model_config, **kwargs):
        """ Read in Zemax ray trace data for a specific LMS configuration (echelle and prism angle, spectral IFU
        selected or not).
        @author: Alistair Glasse
        Read Zemax ray trace data
        """
        Trace.model_config = model_config
        analysis_type, opticon, date_stamp, optical_path_label, coord_in, coord_out = model_config
        self.csv_name = path.split('/')[-1]
        self.is_extended = opticon == Globals.extended
        self.transform_fits_name = None
        self.transforms = []
        self.offset_data = None
        self.a_rms = 0.                          # RMS transform error for trace (microns)
        self.n_mat_terms = None
        silent = kwargs.get('silent', False)
        if not silent:
            print('Reading Zemax model data from ' + path)
        csv_name, lms_config, series = self._read_csv(path, model_config)
        lms_config['opticon'] = opticon
        self.lms_config = lms_config

        # For SPIFU data, spifu slices 1,2 -> echelle order 23, 3,4 _> 24, 5,6 -> 25
        if self.is_extended:
            spifus = series['sp_slice']
            spifu_indices = spifus - 1
            ech_ords = spifu_indices // 2 + 23
            series['ech_ord'] = ech_ords
        self.series = series

        # Count the number of slices and rays
        ech_ords = series['ech_ord']
        self.unique_ech_ords = np.unique(ech_ords)
        slice_nos = series['slice_no']
        self.unique_slices = np.unique(slice_nos)
        spifus = series['sp_slice']
        self.unique_spifu_slices = np.unique(spifus)
        waves = series['wavelength']

        self.wave_reference = self._get_wave_reference()
        self.unique_waves = np.unique(waves)
        self.n_rays, = slice_nos.shape
        self._create_mask(silent)
        do_plot = kwargs.get('do_plot', True)
        self.create_svd_transforms(do_plot=do_plot)
        Trace.create_affine_transforms()
        return

    def _get_wave_reference(self):
        """ Find the reference wavelength for this trace object.  Currently we just take the mean of all ray traced
        wavelengths
        """
        waves = self.series['wavelength']
        wave_ref = np.mean(waves)
        return wave_ref

    def find_wavelength_bounds(self):
        """ Find the wavelength bounds for each slice by back projecting from the detectors.
        """
        affines = Trace.affines
        for transform in self.transforms:
            # Start by finding the detector and row for the slice centre.
            slice_no = transform.slice_configuration['slice_no']
            efp_y = Util.slice_to_efp_y(slice_no, 0.).value
            efp_ref_points = {'efp_x': np.array([0., 0.]), 'efp_y': np.array([efp_y, efp_y]),
                              'efp_w': np.array([self.wave_reference, self.wave_reference])}

            dfp_slice_bs_points = Util.efp_to_dfp(transform, affines, efp_ref_points)
            dfp_y = dfp_slice_bs_points['dfp_y']
            det_nos = [1, 2]
            dfp_x = [0, 2048]
            dfp_points = {'det_nos': np.array(det_nos), 'dfp_x': np.array(dfp_x), 'dfp_y': np.array(dfp_y)}
            efp_points = Util.dfp_to_efp(transform, affines, dfp_points)
            transform.slice_configuration['w_min'] = efp_points['efp_w'][0]
            transform.slice_configuration['w_max'] = efp_points['efp_w'][1]
        return




    def __str__(self):
        fmt = "{:s}, EA={:6.3f} deg, PA={:6.3f} deg, "
        opticon = Trace.model_config[1]
        string = fmt.format(opticon, self.lms_config['ech_ang'], self.lms_config['pri_ang'])
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
        ech_ord = self.unique_ech_ords[0] if opticon == Globals.nominal else self.unique_ech_ords[1]
        slice_filter = {'slice_no': 13, 'spifu_no': spifu_no, 'ech_ord': ech_ord}
        slit_x = self.get_series('ifu_x', slice_filter)
        waves = self.get_series('wavelength', slice_filter)
        wave_bs = np.interp(0.0, slit_x, waves)     # Find wavelength where slit_x == 0.
        pri_ang = self.lms_config['pri_ang']
        return wave_bs, pri_ang

    def create_svd_transforms(self, **kwargs):
        """ Generate transforms to map from phase (wavelength), across-slice coordinates in the entrance focal plane,
        to detector row, column coordinates.
        """
        do_plot = kwargs.get('do_plot', False)
        n_terms = Globals.svd_order
        slice_order = n_terms - 1
        _, _, _, _, fp_in, fp_out = self.model_config
        is_first = True
        a_rms_list = []
        for spifu_no in self.unique_spifu_slices:
            for slice_no in self.unique_slices:
                for ech_ord in self.unique_ech_ords:
                    slice_config = {'slice_no': slice_no, 'spifu_no': spifu_no, 'ech_ord':ech_ord}
                    transform = Transform(trace=self, slice_config=slice_config)
                    waves = self.get_series('wavelength', **slice_config)
                    if len(waves) < 1:      # This happens for mismatched spectral slice and echelle order.
                        continue
                    w_min, w_max = np.amin(waves), np.amax(waves)
                    slice_config['w_min'], slice_config['w_max'] = w_min, w_max

                    n_waves, = waves.shape
                    if n_waves == 0:
                        continue
                    ech_ords = self.get_series('ech_ord', **slice_config)
                    alpha = self.get_series('efp_x', **slice_config)
                    mfp_x = self.get_series(fp_out[0], **slice_config)
                    mfp_y = self.get_series(fp_out[1], **slice_config)

                    phase = Util.waves_to_phases(waves, ech_ords)
                    a, b = Util.solve_svd_distortion(phase, alpha, mfp_x, mfp_y, slice_order, inverse=False)
                    ai, bi = Util.solve_svd_distortion(phase, alpha, mfp_x, mfp_y, slice_order, inverse=True)

                    mfp_x_fit, mfp_y_fit = Util.apply_svd_distortion(phase, alpha, a, b)
                    off_mfp_x, off_mfp_y = mfp_x - mfp_x_fit, mfp_y - mfp_y_fit
                    off_mfp_a = np.sqrt(np.square(off_mfp_x) + np.square(off_mfp_y))
                    a_rms_list.append(off_mfp_a)

                    mats = transform.matrices
                    mats['a'] = a
                    mats['b'] = b
                    mats['ai'] = ai
                    mats['bi'] = bi
                    rays = waves, phase, alpha, mfp_x, mfp_y, mfp_x_fit, mfp_y_fit
                    if do_plot and is_first:        # Plot intermediate and full fit to data
                        fmt = "Distortion residuals, A,B, polynomial fit, SVD cutoff = {:5.1e}\n"
                        tlin1 = fmt.format(Globals.svd_cutoff)
                        self.plot_scatter(transform, rays, plot_correction=True, tlin1=tlin1)
                        is_first = False
                    self.transforms.append(transform)

        a_rms = np.sqrt(np.mean(np.square(np.array(a_rms_list))))
        self.a_rms = a_rms
        return

    def get_series(self, tag, series_filter):
        """ Extract a specific coordinate (identified by 'tag') from the trace for rays which pass through
        a specified spatial and (if the spectral IFU is selected) spectral slice.
        """
        # _, opticon, _, _, _, _ = Trace.model_config
        # opticon = kwargs.get('opticon', Globals.nominal)
        slice_no = series_filter.get('slice_no', 13)
        spifu_no = series_filter.get('spifu_no', 0)
        ech_ord = series_filter.get('ech_ord', None)
        a = self.series[tag]
        slice_nos = self.series['slice_no']
        slice_no_mask = slice_nos == slice_no
        spifu_slices = self.series['sp_slice']
        spifu_no_mask = spifu_slices == spifu_no
        mask = np.logical_and(slice_no_mask, spifu_no_mask)
        if ech_ord is not None:
            ech_ords = self.series['ech_ord']
            ech_ord_mask = ech_ords == ech_ord
            mask = np.logical_and(mask, ech_ord_mask)
        a = np.compress(mask, a)
        return a

    def plot_fit_maps(self, **kwargs):
        """ Plot ray coordinates at the detector for the reference zemax data and also as projected using the
        passed list of transforms (one per slice).
        """
        suppress = kwargs.get('suppress', False)
        if suppress:
            return
        plot = Plot()

        plotdiffs = kwargs.get('plotdiffs', False)
        field = kwargs.get('field', False)
        subset = kwargs.get('subset', False)
        name = self.csv_name
        echelle_angle = self.lms_config['ech_ang']
        fig_title = "{:s} ea = {:4.2f}".format(name, echelle_angle)
        # SPIFU 1 column per spatial slice, 1 column
        n_rows, n_cols = 7, 4
        unique_spifus, unique_slices = self.unique_spifu_slices, self.unique_slices
        spifu_start, slice_start = unique_spifus[0], unique_slices[0]
        if self.is_extended:
            n_rows, = unique_spifus.shape
            n_cols, = unique_slices.shape
        if subset:
            n_rows = 3
            n_cols = 1
        xlim = (None if plotdiffs else [-40.0, 40.0])
        fig, ax_list = plot.set_plot_area(fontsize=14.0, sharex=False, sharey=False,
                                          nrows=1, ncols=1, xlim=xlim)
        ax = ax_list[0, 0]
        fig.suptitle(fig_title)
        for transform in self.transforms:
            slice_config = transform.slice_configuration
            # ech_ord = slice_config['ech_ord']
            slice_no = slice_config['slice_no']
            spifu_no = slice_config['spifu_no']
            ech_ord = slice_config['ech_ord']

            # config = {'slice_no': slice_no, 'spifu_no': spifu_no}
            efp_x = self.get_series('efp_x', slice_no=slice_no, spifu_no=spifu_no, ech_ord=ech_ord)
            efp_y = self.get_series('efp_y', slice_no=slice_no, spifu_no=spifu_no, ech_ord=ech_ord)
            efp_w = self.get_series('wavelength', slice_no=slice_no, spifu_no=spifu_no, ech_ord=ech_ord)
            det_x = self.get_series('det_x', slice_no=slice_no, spifu_no=spifu_no, ech_ord=ech_ord)
            det_y = self.get_series('det_y', slice_no=slice_no, spifu_no=spifu_no, ech_ord=ech_ord)
            efp_points = {'efp_x': np.array(efp_x), 'efp_y': np.array(efp_y), 'efp_w': np.array(efp_w)}
            mfp_fit_points, oob = Util.efp_to_mfp(transform, efp_points)
            det_x_fit, det_y_fit = mfp_fit_points['mfp_x'], mfp_fit_points['mfp_y']
            u, v = det_x - det_x_fit, det_y - det_y_fit
            q = ax.quiver(det_x_fit, det_y_fit, u, v,
                          angles='xy', scale_units='xy', scale=.001,
                          width=0.001)




        # for slice_no in self.unique_slices:



        # for ifu_slice in self.slices:
        #     config, _, rays = ifu_slice
        #     ech_ord, slice_no, spifu_no, _, _ = config
        #     row, col = -1, 0
        #     if subset:
        #         if self.is_extended:
        #             select_row = {6: 0, 3: 1, 1: 2}
        #             if slice_no == 12 and spifu_no in select_row:
        #                 row = select_row[spifu_no]
        #         else:
        #             select_row = {28: 0, 12: 1, 1: 2}
        #         if slice_no in select_row:
        #             row = select_row[slice_no]
        #     else:
        #         spifu_idx = int(spifu_no - spifu_start)
        #         slice_idx = int(slice_no - slice_start)
        #         if self.is_extended:
        #             row = spifu_idx
        #             col = slice_idx
        #         else:
        #             row = int(slice_idx / n_cols)
        #             col = slice_idx % n_cols
        #
        #     if row < 0:       # Skip plot if the pane row is not valid.
        #         continue
        #     ax = ax_list[row, col]
        #     _, _, _, x, y, x_fit, y_fit = rays
        #     if field:                       # Draw filled polygon around field points.
        #         n_uw = len(self.unique_waves)
        #         n_pts = 2 * n_uw + 1        # No. of points on perimeter
        #         xp, yp = np.zeros(n_pts), np.zeros(n_pts)
        #         alphas = rays[2]
        #         uni_alphas = np.unique(alphas)
        #         idxs = np.argwhere(alphas == uni_alphas[0]).flatten()
        #         xp[0:n_uw], yp[0:n_uw] = x[idxs], y[idxs]
        #         idxs = np.argwhere(alphas == uni_alphas[-1]).flatten()
        #         xp[n_uw:-1], yp[n_uw:-1] = np.flip(x[idxs]), np.flip(y[idxs])
        #         xp[-1], yp[-1] = xp[0], yp[0]
        #         ax.fill(xp, yp, color='pink')
        #     if plotdiffs:
        #         u, v = x - x_fit, y - y_fit
        #         q = ax.quiver(x_fit, y_fit, u, v,
        #                       angles='xy', scale_units='xy', scale=.001,
        #                       width=0.001)
        #         if row == 0:
        #             ax.quiverkey(q, X=0.9, Y=1.1, U=0.01, label='10 microns', labelpos='N')
        #     else:
        #         plot.plot_points(ax, x_fit, y_fit, ms=1.0, colour='blue')
        #         plot.plot_points(ax, x, y, ms=1.0, mk='x')
        #     title = "slice {:3.0f}".format(slice_no)
        #     if self.is_extended:
        #         title += ", spectral slice {:3.0f}".format(spifu_no)
        #     ax.set_title(title)
        # ax = ax_list[n_rows-1, 0]
        # ax.set_xlabel('x$_{mfp}$ / mm')
        # ax.set_ylabel('y$_{mfp}$ / mm')

        plot.show()
        return

    def plot_scatter(self, transform, rays, **kwargs):
        """ Plot the spread of offsets between ray trace and fit positions on the detector.
        """
        slice_list = kwargs.get('slice_list', None)
        tlin1_default = "Distortion residuals (Zemax - Zemax Fit): \n"
        tlin1 = kwargs.get('tlin1', tlin1_default)
        slice_config = transform.slice_configuration

        ech_ord = slice_config['ech_ord']
        slice_no = slice_config['slice_no']
        spifu_no = slice_config['spifu_no']
        # Create plot area
        lms_config = transform.lms_configuration
        ech_ang = self.lms_config['ech_ang']
        pri_ang = self.lms_config['pri_ang']
        fmt = "order={:d}, ech. angle={:3.1f}, prism angle={:4.3f}, slice={:d}"
        tlin2 = fmt.format(int(ech_ord), ech_ang, pri_ang, int(slice_no))
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
        gk_text = r'$\sigma$ = ' + '{:4.2f}' + r'$\mu$m'
        fmt = '{:s} residual, ' + gk_text
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
        focal_planes = Trace.spifu_focal_planes if self.is_extended else Trace.nominal_focal_planes

        n_focal_planes = len(focal_planes)
        colour_scheme = kwargs.get('colour_scheme', 'wavelength')
        rgb = self._create_wave_colours(colour_scheme)
        rgb_masked = self._apply_mask(rgb.copy())

        plot = Plot()

        xlabel = 'X [mm]'
        ylabel = 'Y [mm]'
        title = ''
        fig, ax_list = plot.set_plot_area(title=self.__str__(), nrows=1, ncols=n_focal_planes)
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
        csv_name = path.split('/')[-1]
        # name = csv_name.split('.')[0]
        with open(path, 'r') as text_file:
            read_data = text_file.read()
        line_list = read_data.split('\n')
        line_iter = iter(line_list)

        # Read configuration parameters
        efp_map = Trace.spifu_efp_map if self.is_extended else Trace.nominal_efp_map
        efp_x_name = efp_map['efp_x']
        efp_y_name = efp_map['efp_y']
        lms_config = Globals.lms_config_template.copy()
        while True:
            line = next(line_iter)
            tokens = line.split(':')
            if len(tokens) < 2:
                break
            par_to_lms = {'Echelle angle': 'ech_ang',
                          'Prism angle': 'pri_ang'}
            parameter = tokens[0].strip()
            val = float(tokens[1].strip(', '))
            if parameter in par_to_lms.keys():
                lms_name = par_to_lms[parameter]
                lms_config[lms_name] = val
            if parameter == 'Spectral order':
                ech_order = val
        if self.is_extended:            # For SPIFU data, the order is currently calculated from the spifu slice number.
            ech_order = -1

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

        series_names = Trace.series_names
        if self.is_extended:
            series_names.update(Trace.spifu_series_names)
        else:
            series_names.update(Trace.nominal_series_names)

        fp_list = coord_in + coord_out
        for key in fp_list:
            series_names[key] = 'float'
        series = {}
        for name in series_names:
            series[name] = []
        series['ech_ord'] = []    # Add explicitly, since it changes within the csv file for extended data.
        while True:
            line = next(line_iter)
            tokens = line.split(',')
            if len(tokens) < 2:
                break
            for name in series_names:
                if name == 'ech_ord':         # Assign spectral IFU orders later
                    val = 0 if self.is_extended else ech_order
                    series[name].append(val)
                    continue
                if name == 'sp_slice':
                    val = 0
                    if self.is_extended:
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
        int_arrays = ['ech_ord', 'slice', 'spifu_slice']
        for name in series:
            vals = series[name]
            series[name] = np.array(vals, dtype=int) if name in int_arrays else np.array(vals)
        return csv_name, lms_config, series

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
