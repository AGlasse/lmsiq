#!/usr/bin/env python
"""
@author Alistair Glasse
Python object to encapsulate ray trace data for the LMS

18/12/18  Created class (Glasse)
"""
from lms_globals import Globals


class Trace:

    column_dict = {
        'Slice_Number': 0, 'Wavelength': 1,
        'FP2_X': 2, 'FP2_Y': 3, 		# Entrance focal plane
        'FP3_X': 4, 'FP3_Y': 5,			# Slicer
        'FP4_X': 6, 'FP4_Y': 7, 		# IFU
        'FP5_X': 8, 'FP5_Y': 9,			# PDA exit slit
        'FP6_X': 10, 'FP6_Y': 11		# Detector
    }
#    rule_spacing = 18.2					# Echelle rule spacing [um]
    n_axes = len(column_dict)
#    n_slices = 28
#    mm_pix = 0.018
#    pix_edge = 2048
#    det_gap = 3.0				# Gap between active regions of detectors in 2 x 2 mosaic
#    pix_margin = [64, 64]		# Unilluminated margin around all detector (pixels)
#    margin = pix_margin[0] * mm_pix
    echelle_order = -1
    echelle_angle = 0.0
    prism_angle = 0.0
    mean_wavelength = 0.0

    def __init__(self, path, **kwargs):
        """
        @author: Alistair Glasse
        Read Zemax ray trace data
        """
        import numpy as np

        silent = kwargs.get('silent', False)
        if not silent:
            print('Reading Zemax model data from ' + path)
        csv_name = path.split('/')[-1]
        self.name = csv_name.split('.')[0]
        with open(path, 'r') as text_file:
            read_data = text_file.read()
        line_list = read_data.split('\n')
        n_lines = len(line_list)
        n_header = 5

        # Pre-June 2019, the Zemax macro duplicates the last 20 lines.
        n_footer = 0
        if n_lines > 11206:
            n_footer = 20
        self.echelle_order = self._parse_header_line(line_list[0])
        self.echelle_angle = self._parse_header_line(line_list[1])
        self.prism_angle = self._parse_header_line(line_list[2])
        n_records = n_lines - n_header - n_footer - 1
        new_data = np.zeros((self.n_axes, n_records))
        for j in range(0, n_records):
            record = line_list[j + n_header]
            token = record.split(',')
            for i in range(0, self.n_axes):
                new_data[i,j] = float(token[i])
        self._set_data(new_data, silent)
        self.n_rays = n_records
        self.n_rays_slice = int(n_records / Globals.n_slices)
        return

    def to_transforms(self, n_terms):
        from lms_util import Util
        util = Util()

#		print("trace.to_transform - Creating fixed affine 'AB' transforms (Phase,Alpha) -> (DET_X,DET_Y)")
        tf_abs = []
        tf_order = n_terms - 1
        for s in range(0, Globals.n_slices):
            sno = s + 1
            phase = self.get('Phase', slice=sno)
            alpha = self.get('FP2_X', slice=sno)
            det_x = self.get('FP6_X', slice=sno)
            det_y = self.get('FP6_Y', slice=sno)

            (a, b) = util.distortion_fit(phase, alpha, det_x, det_y, tf_order, inverse=False)
            (ai, bi) = util.distortion_fit(phase, alpha, det_x, det_y, tf_order, inverse=True)

            tf_ab = (s, a, b, ai, bi)
            tf_abs.append(tf_ab)
        return tf_abs

    def apply_fwd_transforms(self, tf_list):
        from lms_util import Util

        util = Util()
        for tf in tf_list:
            (s, a, b, ai, bi) = tf
            sno = s + 1
            phase = self.get('Phase', slice=sno)
            alpha = self.get('FP2_X', slice=sno)
            ifp_x_proj, ifp_y_proj = util.apply_distortion(phase, alpha, a, b)
            det_x_proj, det_y_proj = util.apply_distortion(ifp_x_proj, ifp_y_proj, c, d)
        return det_x_proj, det_y_proj

    def get_fit_statistics(self, x, y, det_x, det_y):
        from lms_util import Util
        import numpy as np

        util = Util()

        dis_x_list = []
        dis_y_list = []
        dis_x = x - det_x
        dis_y = y - det_y
        dis_x_list.append(dis_x)
        dis_y_list.append(dis_y)
        dx = np.asarray(dis_x_list)
        dy = np.asarray(dis_y_list)
        ave_x = np.mean(dx)
        ave_y = np.mean(dy)
        rms_x = np.std(dx)
        rms_y = np.std(dy)
        return ave_x, rms_x, ave_y, rms_y

    def get_transform_offsets(self, tf_list):
        from lms_util import Util
        import numpy as np

        util = Util()
        offset_x = np.zeros((Globals.n_slices, self.n_rays_slice))
        offset_y = np.zeros((Globals.n_slices, self.n_rays_slice))

        for tf in tf_list:
            (s, a, b, ai, bi) = tf
            sno = s + 1
            phase = self.get('Phase', slice=sno)
            alpha = self.get('FP2_X', slice=sno)
            det_x = self.get('FP6_X', slice=sno)
            det_y = self.get('FP6_Y', slice=sno)
            det_x_fit, det_y_fit = util.apply_distortion(phase, alpha, a, b)
            offset_x[s] = det_x_fit - det_x
            offset_y[s] = det_y_fit - det_y
        return offset_x, offset_y

    def plot_fit_maps(self, offset_x, offset_y, **kwargs):
        """ Plot ray coordinates at detector for the reference zemax data and also
        as projected using the passed list of transforms (one per slice). """
        from lmsplot import LmsPlot
        from utils import Utilities
        plot = LmsPlot()
        util = Utilities()

        suppress = kwargs.get('suppress', False)
        if (suppress):
            return

        plotdiffs = kwargs.get('plotdiffs', False)
        fig_title = "{:s} ea = {:4.2f}".format(self.name, self.echelle_angle)
        nrows = 7
        ncols = 4
        xlim = (None if plotdiffs else [-60.0, 60.0])
        ax_list = plot.set_plot_area(fig_title, nrows=nrows, ncols=ncols, xlim=xlim)
        for s in range(0, Globals.n_slices):
            sno = s + 1
            det_x = self.get('FP6_X', slice=sno)
            det_y = self.get('FP6_Y', slice=sno)
            det_x_fit = det_x + offset_x
            det_y_fit = det_y + offset_y
            row = s % nrows
            col = int(s / nrows)
            ax = ax_list[row, col]
            plot.plot_points(ax, det_x_fit, det_y_fit, ms=1.0, colour='blue')
            plot.plot_points(ax, det_x, det_y, ms=1.0, mk='x')
        plot.show()
        return

    def _set_data(self, new_data, silent):
        """ Copy a new data array into this trace object and recalculate the
        wavelength coverage information
        :param new_data:
        :return:
        """
        self.data = new_data
        self._update_mask(silent)
        self._update_wavelength()
        return

    def _parse_header_line(self, line):
        tokens = line.split(' ')
        n_tokens = len(tokens)
        val = float(tokens[n_tokens-1])
        return val

    def get(self, tag, **kwargs):
        import numpy as np

        slice = kwargs.get('slice', None)
        if tag == 'Phase':
            col_idx = self.column_dict.get("Wavelength")
            w = self.data[col_idx]
            a = w * self.echelle_order / (2.0 * Globals.rule_spacing)
        else:
            col_idx = self.column_dict.get(tag)
            a = self.data[col_idx]
        if slice is not None:
            slice_array = self.data[0]
            idx = (slice_array == slice)
            a = np.compress(idx, a)
        return a

    def _update_mask(self, silent):
        import numpy as np

        edge = Globals.pix_edge * Globals.mm_lmspix
        xy_f = (Globals.det_gap / 2.0) + edge - Globals.margin
        xy_n = (Globals.det_gap / 2.0) + Globals.margin
        xy_bounds = [[-xy_f, -xy_n, -xy_f, -xy_n],
                    [-xy_f, -xy_n,  xy_n,  xy_f],
                    [xy_n,  xy_f, -xy_f, -xy_n],
                    [xy_n,  xy_f,  xy_n,  xy_f]]

        x = self.get('FP6_X')
        y = self.get('FP6_Y')

        nPts = len(x)
        mask = np.zeros(nPts, dtype=np.int)	# >0 = Ray hits a detector
        for i in range(0, nPts):
            for j in range(0, 4):
                xyB = xy_bounds[j]
                isInX = x[i] > xyB[0] and x[i] < xyB[1]
                isInY = y[i] > xyB[2] and y[i] < xyB[3]
                isHit = isInX and isInY
                detNumber = j + 1
                mask[i] = detNumber if isHit else mask[i]
        nHits = np.count_nonzero(mask)
        if not silent:
            fmt = 'Rays hitting any detector = {:10d} / {:10d}'
            print(fmt.format(nHits, nPts))
        self.mask = mask
        return

    def _update_wavelength(self):
        import numpy as np
        import math

        wav = self.get('Wavelength')
        slices = self.get('Slice_Number')
        nPts = len(wav)
        rgb = np.zeros((3, nPts))

        wMin = np.amin(wav)
        wMax = np.amax(wav)
        rMin = 0.0
        rMax = 1.0
        bMin = 1.0
        bMax = 0.0

        sMin = int(min(slices))
        sMax = int(max(slices))
        nSlices = sMax - sMin + 1
        self.wSlice = np.zeros((4, nSlices))
        self.wSlice[0,:] =  100.
        self.wSlice[1,:] = -100.
        self.wSlice[2,:] =  100.
        self.wSlice[3,:] = -100.
        for i in range(0, nPts):
            w = wav[i]
            s = int(slices[i]) - sMin
            if self.mask[i] > 0:
                d = self.mask[i]		# Detector number (1, 2, 3, 4)
                if d == 1 or d == 2:
                    self.wSlice[0,s] = w if w < self.wSlice[0,s] else self.wSlice[0,s]
                    self.wSlice[1,s] = w if w > self.wSlice[1,s] else self.wSlice[1,s]
                else:
                    self.wSlice[2,s] = w if w < self.wSlice[2,s] else self.wSlice[2,s]
                    self.wSlice[3,s] = w if w > self.wSlice[3,s] else self.wSlice[3,s]

                f = (w - wMin) / (wMax - wMin)
                r = rMin + f * (rMax - rMin)
                g = math.sin(f * math.pi)
                b = bMin + f * (bMax - bMin)
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

    def plot_slice_map(self, sno, **kwargs):
        """ Plot the planes in FP2 (nxlambda/2d, y) and at the detector (x, y)
        which are used to find the transforms """
        from lmsplot import LmsPlot
        import numpy as np

        suppress = kwargs.get('suppress', False)
        if (suppress):
            return

        plot = LmsPlot()
        titles = ['LMS EFP', 'Detector']

        wavel = self.get('Wavelength', slice=sno)
        phase = wavel * self.echelle_order / (2.0 * Globals.rule_spacing)
        fp2_x = self.get('FP2_X', slice=sno)
        det_x = self.get('FP6_X', slice=sno)
        det_y = self.get('FP6_Y', slice=sno)

        xlabels = ['Phase (n lambda / 2 d)', 'X [mm]']
        ylabels = ['alpha [mm]', 'Y [mm]']
        fig_title = "{:s}, slice= {:d}".format(self.name, sno)
        ax_list = plot.set_plot_area(fig_title, nrows=1, ncols=2)

        xs = [phase, det_x]
        ys = [fp2_x, det_y]

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

    def _find_limits(self, a, margin):
        amin = min(a)
        amax = max(a)
        arange = amax - amin
        amargin = margin * arange
        limits = [amin - amargin, amax + amargin]
        return limits


    def plot(self, **kwargs):
        """ Plot coordinates at all focal surfaces. FP2_X v FP2_Y, FP4_X v Y

        :param kwargs:
        :return:
        """
        from lmsplot import LmsPlot
        import numpy as np

        suppress = kwargs.get('suppress', False)
        if (suppress):
            return

        plot = LmsPlot()
        titles = ['LMS EFP', 'IFU exit', 'PDA exit', 'Detector']
        xidx = [2, 6, 8, 10]
        yidx = [3, 7, 9, 11]
        xlims = [[-2.0, 2.0], [-0.05, 0.05], [-0.2, 2.0], [-40.0, 40.0]]
        ylims = [[-1.5, 1.5], [-40.0, 40.0], [-42.0, 42.0], [-40.0, 40.0]]
        isdispersed = [False, False, True, True]
        xlabel = 'X [mm]'
        ylabel = 'Y [mm]'
        black = np.zeros((3, self.n_rays))
        ax_list = plot.set_plot_area(self.name, nrows=2, ncols=2)
        idx = 0
        for i in range(0, 2):
            for j in range(0, 2):
                ax = ax_list[i, j]
                x = self.data[xidx[idx]]
                y = self.data[yidx[idx]]
                rgb = (self.rgb if isdispersed[idx] else black)
                ax.set_title(titles[idx], loc='left')
                ax.set_xlim(xlims[idx])
                ax.set_ylim(ylims[idx])
                if i == 1:
                    ax.set_xlabel(xlabel)
                if j == 0:
                    ax.set_ylabel(ylabel)
                print('Done {:d} of 4'.format(idx))
                plot.plot_points(ax, x, y, fs='full', ms=1.0, mk='o', rgb=rgb)
                idx = idx + 1
        plot.show()
        return

    def _get_parameters(self):
        import numpy as np

        ea = self.echelle_angle
        so = self.echelle_order
        pa = self.prism_angle
        w1 = np.max(self.wSlice[0,:])
        w2 = np.min(self.wSlice[1,:])
        w3 = np.max(self.wSlice[2,:])
        w4 = np.min(self.wSlice[3,:])
        return (ea, so, pa, w1, w2, w3, w4)

    def to_string(self):
        """ Get single comma delimited text string representation of instrument
        configuration and wavelength coverage.
        :return:
        """
        (ea, so, pa, w1, w2, w3, w4) = self._get_parameters()
        fmt = '{:6.3f},{:3.0f},{:6.3f},{:8.3f},{:8.3f},{:8.3f},{:8.3f}'
        text = fmt.format(ea, so, pa, w1, w2, w3, w4)
        return text

    def tfs_to_text(self, tfs):
        from lms_util import Util

        util = Util()
        shape = tfs[0][1].shape
        n_terms = shape[0]

        n_slices = len(tfs)
        text = ''
        for s in range(0, n_slices):
            tf = tfs[s]
            (s, a, b, ai, bi) = tf
            sno = s + 1
            phase = self.get('Phase', slice=sno)
            alpha = self.get('FP2_X', slice=sno)
            x_det = self.get('FP6_X', slice=sno)
            y_det = self.get('FP6_Y', slice=sno)

            x, y = util.apply_distortion(phase, alpha, a, b)
            (ave_x, rms_x, ave_y, rms_y) = self.get_fit_statistics(x, y, x_det, y_det)
            fit_stats_string = ",{:8.3f},{:8.3f},{:8.3f},{:8.3f}\n".format(ave_x, rms_x, ave_y, rms_y)
            for i in range(0, 4):
                mat = tf[i + 1]
                for j in range(0, n_terms):
                    line = self.to_string()
                    line = line + ',{:3d},{:3d},{:3d}'.format(s, i, j)
                    fmt = ',{:16.7e}'
                    for k in range(0, n_terms):
                        line = line + fmt.format(mat[j, k])
                    line = line + fit_stats_string
                    text = text + line
        return text

    def print(self):
        """ Print trace parameters, instrument configuration and wavelength
        coverage.
        :return:
        """
        (ea, so, pa, w1, w2, w3, w4) = self._get_parameters()
        fmt = 'Configuration: ech_ang ={:10.3f}, prism_angle ={:10.3f}'
        print(fmt.format(ea, pa))
        fmt = 'Wavelength coverage = {:10.3f}{:10.3f}{:10.3f}{:10.3f}'
        print(fmt.format(w1, w2, w3, w4))
        return
