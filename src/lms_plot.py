#!/usr/bin/env python
""" Created on Feb 21, 2018

@author: achg
"""
import numpy as np
import matplotlib.pyplot as plt


class Plot:

    def __init__(self):
        return

    def set_plot_area(self, title, **kwargs):
        figsize = kwargs.get('figsize', [12, 9])
        xlim = kwargs.get('xlim', None)            # Common limits for all plots
        ylim = kwargs.get('ylim', None)            # Common limits for all plots
        xlabel = kwargs.get('xlabel', '')          # Common axis labels
        ylabel = kwargs.get('ylabel', '')
        ncols = kwargs.get('ncols', 1)             # Number of plot columns
        nrows = kwargs.get('nrows', 1)
        remplots = kwargs.get('remplots', None)
        aspect = kwargs.get('aspect', 'auto')      # 'equal' for aspect = 1.0
        fontsize = kwargs.get('fontsize', 16)

        plt.rcParams.update({'font.size': fontsize})

        sharex = xlim is not None
        sharey = ylim is not None
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
                ax_list[rps[i,0], rps[i,1]].remove()
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

    def wavelengths_v_ech_ang(self, configs, **kwargs):

        suppress = kwargs.get('suppress', False)
        if suppress:
            return

        (eas, eos, pas, w1s, w2s, w3s, w4s) = configs

        n_configs = len(eas)
        xlim = [2.5, 5.5]
        ylim = [5.0, 7.5]
        ax_list = self.set_plot_area('Zemax vXX.YY',
                                     xlim=xlim, xlabel='Wavelength [um]',
                                     ylim=ylim, ylabel='Prism angle + 0.02 x Ech. angle')
        ax = ax_list[0, 0]
        y = pas + 0.02 * eas
        for i in range(0, n_configs):
            self.plot_line(ax, [w1s[i], w4s[i]], [y[i],  y[i]], fs='full', ms=2.0)

        self.show()
        return

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
            ax.plot(x, y, clip_on=True, ls='-', lw=5.0, color=col)
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

    def config_v_coeffs(self, eo, sno, mat, x, coeffs, polys, **kwargs):
        suppress = kwargs.get('suppress', False)
        if suppress:
            return
        cs = coeffs.shape
        n_terms = cs[0]
        remplots = []
        for i in range(1, n_terms):
            for j in range(n_terms-i, n_terms):
                remplots.append([i, j])
        xmin = min(x)
        xmax = max(x)
        xlim = self._auto_lim(x, 0.1)
        ylim = [0.0, 2.0]
        xt, yt = self._get_text_position(xlim, ylim, pos='tl', inset=[0.03, 0.17])
        n_fit_points = 20
        xfit = xmin + np.asarray(range(0, n_fit_points+1)) * (xmax - xmin) / float(n_fit_points)

        matlabels = ["A", "B", "AI", "BI"]
        fmt = "Echelle order {:d}, Slice {:d}, {:s}[i,j]"
        title = fmt.format(eo, sno, matlabels[mat])
        ax_list = self.set_plot_area(title,
                                     xlim=xlim, xlabel="Ech angle / deg.",
                                     ylim=ylim, ylabel="Normalised coeff.",
                                     ncols=n_terms, nrows=n_terms, remplots=remplots)
        for i in range(0, n_terms):
            for j in range(0, n_terms-i):
                ax = ax_list[i, j]
                y = coeffs[i, j]
                ymean = np.nanmean(y)
                ynorm = np.divide(y, ymean)
                label = "[{:d},{:d}] {:7.1e}".format(i, j, ymean)
                ax.text(xt, yt, label)
                self.plot_points(ax, x, ynorm, fs='full', ms=3.0)
                yfit = np.polyval(polys[i,j], xfit) / ymean
                self.plot_line(ax, xfit, yfit, colour='red')
        self.show()

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

#    @staticmethod
    def plot_points(self, ax, x, y, **kwargs):
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

    def plotCoverage(self, pars, **kwargs):
        nConfigs = len(pars)
        wLim = kwargs.get('wlim', [2.7, 5.7])
        paLim = kwargs.get('ylim', [5.0, 7.5])

        self.setPlotArea('Prism angle v Wavelength Coverage',
                         wLim, 'Wavelength [micron]',
                         paLim, 'Prism angle [deg]')
        ax = self.axList[0, 0]
        eaMin = -7.0
        eaMax =  7.0
        for i in range(0, nConfigs):
            (ea, so, pa, w1, w2, w3, w4) = pars[i]
            f = (ea - eaMin) / (eaMax - eaMin)
            r = 0.2
            g = f
            b = 1.0 - f
            print(r, g, b)
            ax.plot([w1,w4], [pa,pa],
                    color=[r,g,b], linestyle='-', linewidth=2.0)
        self.show()

    def show(self):
        """ Wrapper for matplotlib show function. """
        import matplotlib.pyplot as plt
        plt.show()
