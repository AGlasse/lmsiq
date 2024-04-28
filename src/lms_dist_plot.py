#!/usr/bin/env python
""" Created on Feb 21, 2018

@author: achg
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from lms_globals import Globals


class Plot:

    def __init__(self):
        return

    @staticmethod
    def set_plot_area(title, **kwargs):
        figsize = kwargs.get('figsize', [9, 6])
        xlim = kwargs.get('xlim', None)            # Common limits for all plots
        ylim = kwargs.get('ylim', None)            # Common limits for all plots
        xlabel = kwargs.get('xlabel', '')          # Common axis labels
        ylabel = kwargs.get('ylabel', '')
        ncols = kwargs.get('ncols', 1)             # Number of plot columns
        nrows = kwargs.get('nrows', 1)
        remplots = kwargs.get('remplots', None)
        aspect = kwargs.get('aspect', 'auto')      # 'equal' for aspect = 1.0
        fontsize = kwargs.get('fontsize', 12)

        plt.rcParams.update({'font.size': fontsize})

        sharex = kwargs.get('sharex', False)
        sharey = kwargs.get('sharey', False)
        fig, ax_list = plt.subplots(nrows, ncols, figsize=figsize,
                                    sharex=sharex, sharey=sharey,
                                    squeeze=False)
        fig.patch.set_facecolor('white')
        fig.suptitle(title)

        for i in range(0, nrows):
            for j in range(0, ncols):
                ax = ax_list[i,j]
                ax.set_aspect(aspect)       # Set equal axes
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                if (i == nrows-1 and j == 0):
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
        if remplots is not None:
            rps = np.atleast_2d(remplots)
            for i in range(0, len(rps)):
                ax_list[rps[i,0], rps[i, 1]].remove()
        return ax_list

    def pixel_v_mech_rot(self, coverage, **kwargs):
        suppress = kwargs.get('suppress', False)
        if suppress:
            return
        eas, pas, w1max, w2min, w3max, w4min = coverage
        n_pix_21 = 2048
        n_configs = len(eas)
        xlim = [2.5, 5.5]
        ylim = [-2.0, 2.0]
        ax_list = self.set_plot_area('Zemax vXX.YY',
                                     xlim=xlim, xlabel='Wavelength [um]',
                                     ylim=ylim, ylabel='Mech. rot. / Image motion [arcsec/pixel]')
        ax = ax_list[0, 0]

        ea_settings = np.unique(eas)
        for ea_setting in ea_settings:
            idx = np.where(eas == ea_setting)
            pas_vals = pas[idx]
            w2min_vals = w2min[idx]
            w1max_vals = w1max[idx]
            n_vals = len(pas_vals)
            dw2 = w2min_vals[1:n_vals] - w2min_vals[0:n_vals-1]
            dw_det = w2min_vals[1:n_vals] - w1max_vals[1:n_vals]
            dpix = n_pix_21 * dw2 / dw_det
            drot = (pas_vals[1:n_vals] - pas_vals[0:n_vals-1]) * 3600.0
            x = w2min_vals[1:n_vals]
            y = drot / dpix
            self.plot_points(ax, x, y)
        self.show()

    def efficiency_v_wavelength(self, weoas, **kwargs):

        waves, effs, orders, angles = weoas
        n_angs, n_orders = waves.shape
        xlim_default = [np.min(waves), np.max(waves)]
        ylim_default = [0.0, 1.0]
        xlim = kwargs.get('xlim', xlim_default)
        ylim = kwargs.get('ylim', ylim_default)
        xcoverage = xlim[1] - xlim[0]
        xtick_spacing = 0.05 if xcoverage < 1.0 else 0.2
        ax_list = self.set_plot_area('Echelle eficiency',
                                     xlim=xlim, xlabel='Wavelength [um]',
                                     ylim=ylim, ylabel='Efficiency')
        ax = ax_list[0, 0]
        xtick_vals = np.arange(xlim[0], xlim[1], xtick_spacing)
        xtick_labels = []
        for v in xtick_vals:
            xtl = "{:10.2f}".format(v)
            xtick_labels.append(xtl)
        ax.set_xticks(xtick_vals)
        ax.set_xticklabels(xtick_labels)
        colours = Plot._make_colours(n_orders)
        for i in range(0, n_orders):
            order = orders[i]
            x = waves[:, i]
            y = effs[:, i]
            col = colours[i]
            ax.plot_focal_planes(x, y, clip_on=True, ls='-', lw=5.0, color=col)
            jmid = int(n_angs / 2)
            xt, yt = x[jmid], y[jmid]
            ax.text(xt, yt, "{:3d}".format(order), color=col, ha='left', va='bottom')
        self.show()
        return

    @staticmethod
    def _auto_lim(a, margin):
        amin = min(a)
        amax = max(a)
        arange = amax - amin
        amargin = margin * arange
        lim = [amin - amargin, amax + amargin]
        return lim

    @staticmethod
    def _make_colours(n_colours):
        """ Generate a list of colours """
        colours = []
        r, g, b = 0.0, 0.0, 0.0         # Always start with black
        for i in range(0, n_colours):
            r += 0.9
            g += 0.3
            b += 0.5
            r = r if r < 1.0 else r - 1.0
            g = g if g < 1.0 else g - 1.0
            b = b if b < 1.0 else b - 1.0
            colours.append([r, g, b])
        return colours

    @staticmethod
    def _get_text_position(xlim, ylim, **kwargs):
        pos = kwargs.get('pos', 'tl')
        inset = kwargs.get('inset', [0.1, 0.1])
        posdict = {'tl': 0, 'tr': 1}

        xr = xlim[1] - xlim[0]
        xt = xlim[0] + inset[0] * xr
        yr = ylim[1] - ylim[0]
        yt = ylim[1] - inset[1] * yr

        return xt, yt

    @staticmethod
    def matrix_fit(x, matrices, fit_matrices):
        n_mat, n_terms, _ = matrices.shape
        ax_list = Plot.set_plot_area('title',
                                     xlabel='Prism angle [deg]',
                                     ylabel="Matrix term",
                                     ncols=n_terms, nrows=n_terms)
        for row in range(0, n_terms):
            for col in range(0, n_terms):
                ax = ax_list[row, col]
        return

    @staticmethod
    def round_trip(y_arr, y_rtn_arr, **kwargs):
        title = kwargs.get('title', 'title')
        ax_list = Plot.set_plot_area(title,
                                     xlabel="sample",
                                     ylabel="val")
        ax = ax_list[0, 0]
        y = y_arr.flatten()
        n_samples, = y.shape
        y_rtn = y_rtn_arr.flatten()
        x = np.arange(0, n_samples, 1)
        ax.plot(x, y,
                color='black', clip_on=True,
                fillstyle='none', marker='o', mew=1.0, ms=3.0, linestyle='None')
        ax.plot(x, y_rtn,
                color='red', clip_on=True,
                fillstyle='none', marker='x', mew=1.0, ms=3.0, linestyle='None')
        Plot.show()
        return

    @staticmethod
    def wavelength_coverage(traces, optical_configuration):
        ax_list = Plot.set_plot_area('Wavelength coverage',
                                     xlabel="Wavelength / micron",
                                     ylabel="Prism angle + 0.02 x Echelle angle ")
        ax = ax_list[0, 0]
        ccs4_colours = mcolors.CSS4_COLORS
        if optical_configuration == Globals.spifu:
            ccs4_colours = mcolors.TABLEAU_COLORS

        colours = sorted(ccs4_colours,
                         key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))),
                         reverse=True)
        colour_iterator = iter(colours)
        config_colour = {}
        old_labels = []

        for trace in traces[0:-1]:
            ech_angle, prism_angle = trace.parameter['Echelle angle'], trace.parameter['Prism angle']
            tag = "{:5.2f}{:5.2f}".format(ech_angle, prism_angle)
            config_colour[tag] = next(colour_iterator, 'black')
            for tf in trace.slice_objects:
                config, matrices, offset_corrections, rays, wave_bounds = tf
                label, slice_no, spifu_no = config
                x = rays[0]
                y = prism_angle + 0.1 * ech_angle + 0.004 * spifu_no + 0.001 * rays[4]
                x_label, y_label = x[0], y[0]

                tag = "{:5.2f}{:5.2f}".format(ech_angle, prism_angle)
                colour = config_colour[tag]
                ax.plot(x, y, color=colour, clip_on=True,
                        fillstyle='none', marker='.', mew=1.0, ms=3, linestyle='None')

                label = "{:d}".format(int(label))
                if spifu_no != -1:
                    label += "/{:d}".format(int(spifu_no))
                is_new_label = label not in old_labels
                # if is_new_label:
                #     ax.text(x_label, y_label, label)
                #     old_labels.append(label)
        Plot.show()
        return

    @staticmethod
    def plot_points(ax, x, y, **kwargs):
        """ Plot an array of points in the open plot region. """

        n_pts = len(x)
        fs = kwargs.get('fs', 'none')
        mk = kwargs.get('mk', 'o')
        mew = kwargs.get('mew', 1.0)
        ms = kwargs.get('ms', 3)
        colour = kwargs.get('colour', 'black')
        rgb = kwargs.get('rgb', None)
        if rgb is None:
            ax.plot(x, y, color=colour, clip_on=True,
                    fillstyle=fs, marker=mk, mew=mew, ms=ms, linestyle='None')
        else:
            for i in range(0, n_pts):
                ax.plot(x[i], y[i], color=rgb[:, i], clip_on=True,
                        fillstyle=fs, marker=mk, mew=mew, ms=ms)
        return

    def plot_line(self, ax, x, y, **kwargs):
        colour = kwargs.get('colour', 'black')
        ls = kwargs.get('linestyle', '-')
        lw = kwargs.get('linewidth', 1.0)

        ax.plot(x, y, clip_on=True, linestyle=ls, linewidth=lw, color=colour)
        return

    def plot_point(self, ax, xy, **kwargs):
        """ Plot a single point in the open plot region.
        """
        import numpy as np
        xyList = np.array([[xy[0]], [xy[1]]])
        self.plot_points(ax, xyList, **kwargs)

    @staticmethod
    def show():
        """ Wrapper for matplotlib show function. """
        import matplotlib.pyplot as plt
        plt.show()
