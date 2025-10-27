#!/usr/bin/env python
""" Created on Feb 21, 2018

@author: achg
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from lms_globals import Globals


class Plot:

    def __init__(self):
        return

    @staticmethod
    def set_plot_area(**kwargs):
        figsize = kwargs.get('figsize', [9, 6])
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

        sharex = kwargs.get('sharex', False)
        sharey = kwargs.get('sharey', False)
        fig, ax_list = plt.subplots(nrows, ncols, figsize=figsize,
                                    sharex=sharex, sharey=sharey,
                                    squeeze=False)
        fig.patch.set_facecolor('white')

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
        return fig, ax_list

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
    def make_rgb_gradient(vals):
        n_pts = len(vals)
        rgb = np.zeros((n_pts, 3))

        w_min = np.amin(vals)
        w_max = np.amax(vals)
        r_min, r_max = 0., 1.
        b_min, b_max = 1., 0.

        f = (vals - w_min) / (w_max - w_min)
        rgb[:, 0] = r_min + f * (r_max - r_min)
        rgb[:, 1] = np.sin(f * math.pi)
        rgb[:, 2] = b_min + f * (b_max - b_min)
        return rgb

    @staticmethod
    def wave_v_prism_angle(wpa_fit, wpa_model, wave_boresights, prism_angles, differential=False):
        """ Plot fit of prism angle to wavelength compared with the trace data it is derived from.
        """
        x, y = np.array(wave_boresights), np.array(prism_angles)
        if differential:
            dp = y[1:] - y[:-1]
            dw = x[1:] - x[:-1]
            x = x[1:]
            y = dp / dw

        fig, [ax1, ax2] = plt.subplots(figsize=(10, 8), ncols=1, nrows=2,
                                       sharex=True)
        ylabel = 'Prism rotation sensitivity micron / deg' if differential else 'Prism rotation angle / deg'
        xlabel = "Wavelength / $\mu$m"

        coeffs = wpa_fit['wpa_opt']
        y_fit = wpa_model(x, *coeffs)
        form_text = ", $\phi_{prism} = $"
        lam_text = ''
        for i, coeff in enumerate(coeffs):
            if i > 1:
                lam_text = r'$\lambda^{' + str(i) + r'}$'
            form_text += "{:+6.3f}{:s}".format(coeff, lam_text) # + plus_text
            lam_text = '$\lambda$'

        title1 = 'Prism dispersion data and best fit' + form_text
        ax1.set_title(title1, loc='left')
        ax1.set_ylabel(ylabel)
        ax1.plot(x, y, color='blue', clip_on=True,
                 fillstyle='full', marker='+', ms=5., linestyle='None')
        if not differential:
            ax1.plot(x, y_fit, color='black', clip_on=True, lw=1., linestyle='solid')
        ax1.grid(True)

        ax2.set_title('Residual (data - fit)', loc='left')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel(ylabel)
        y_residual = y - y_fit
        ax2.plot(x, y_residual, color='blue', clip_on=True,
                 fillstyle='full', marker='+', ms=5., linestyle='None')

        ax2.grid(True)
        plt.show()
        return

    @staticmethod
    def wxo_fit(wxo_fit, term_values, surface_model, plot_residuals=False):
        """ Plot the polynomial surface fit to wavelength x echeller order as a function of echelle and
        prism rotation angle.
        """
        x = np.array(term_values['pri_ang'])
        y = np.array(term_values['ech_ang'])
        wxo = np.array(term_values['w_bs']) * np.array(term_values['ech_orders'])
        wxo_opt = wxo_fit['wxo_opt']       # Fit parameters
        wxo_model = surface_model((x, y), *wxo_opt)
        f_resid = (wxo - wxo_model) * 1000  # Fractional residual (for fit points)
        f_res_std = np.std(f_resid)
        print("Residual stdev = {:7.3f} nm".format(f_res_std))
        z = wxo
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8),
                               ncols=1, nrows=1,
                               sharex=True, sharey=True)

        zlabel = 'n $\lambda$ / micron'
        suptitle = r'$n \lambda = f(\phi_{pri}, \psi_{ech})$'
        if plot_residuals:
            suptitle = 'Residuals ' + suptitle
            z = f_resid
            zlabel = 'n $\lambda$ residual / nano-m'
        fig.suptitle(suptitle)
        ax.scatter(x, y, z, color='blue')
        ax.set_xlabel('prism angle, $\phi_{pri}$ / deg')
        ax.set_ylabel('echelle angle, $\psi_{ech}$ / deg')
        ax.set_zlabel(zlabel)

        if not plot_residuals:
            x_range = np.linspace(5, 8., 50)
            y_range = np.linspace(-6., 6., 50)
            xy_grid = np.meshgrid(x_range, y_range)
            z_grid = surface_model(xy_grid, *wxo_opt)
            ax.plot_surface(xy_grid[0], xy_grid[1], z_grid, color='red', alpha=0.5)
        plt.show()
        return

    @staticmethod
    def transform_fit(term_fit, term_values, surface_model, plot_residuals=False, do_plots=True):
        """ Make a 3D plot of data points and the surface that fits them.
        """
        pri_ang = np.array(term_values['pri_ang'])
        ech_ang = np.array(term_values['ech_ang'])
        matrices = term_values['matrices']
        slice_no = term_values['slice_no']
        x, y = pri_ang, ech_ang

        svd_order = Globals.svd_order
        fig, ax_list = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 10),
                                    ncols=svd_order, nrows=svd_order,
                                    sharex=True, sharey=True)
        for row in range(0, svd_order):
            for col in range(0, svd_order):
                if row + col > svd_order - 1:
                    ax_list[row, col].remove()

        st1 = r'$c_{i,j} = g_{i,j}(\phi_{pri}, \psi_{ech})$'
        st2 = '\nTransform fit coefficients $c_{row, col}$'
        st3 = " for slice no.= {:d}".format(slice_no)
        zlabel = 'coefficient value'
        if plot_residuals:
            st2 = '\nTransform fit coefficient fractional residuals '
            zlabel = 'residual / %'
        suptitle = st1 + st2 + st3
        fig.suptitle(suptitle)

        f_residuals = {'slice_no': slice_no}
        for mat_tag in ['a']:
            f_residuals[mat_tag] = {}
            for row in range(0, svd_order):
                f_residuals[mat_tag][row] = {}
                for col in range(0, svd_order):
                    if row + col > svd_order - 1:
                        continue
                    z_term = []
                    for matrix in matrices:
                        mat = matrix[mat_tag]
                        z_term.append(mat[row, col])
                    z = np.array(z_term)
                    mat = term_fit[mat_tag]
                    term_opt = mat[row, col]
                    z_model = surface_model((x, y), *term_opt)
                    f_resid = 100. * (z - z_model) / z_model       # Fractional residual (for fit points) / %
                    f_residuals[mat_tag][row][col] = f_resid
                    ax = ax_list[row, col]
                    title = "c[{:d}, {:d}]".format(row, col)
                    ax.set_title(title)
                    ax.xaxis.set_major_formatter("{x:1.2f}")
                    if row == 2 and col == 1:
                        ax.set_xlabel('$\phi_{pri}$ / deg')
                        ax.set_ylabel('$\psi_{ech}$ / deg')
                        ax.set_zlabel(zlabel)
                    if plot_residuals:
                        ax.scatter(x, y, f_resid, color='black', alpha=0.5)
                        continue
                    x_range = np.linspace(5, 8., 50)
                    y_range = np.linspace(-6., 6., 50)
                    xy_grid = np.meshgrid(x_range, y_range)
                    z_grid = surface_model(xy_grid, *term_opt)
                    ax.scatter(x, y, z, color='red', alpha=0.5)
                    ax.plot_surface(xy_grid[0], xy_grid[1], z_grid, color='grey', alpha=0.5)
        Plot.show()
        return f_residuals

    @staticmethod
    def mfp_projections(mfp_projections, trace):

        n_ax_rows = 0
        for slice_no in mfp_projections.keys():
            mfp_slice = mfp_projections[slice_no]
            for spifu_no in mfp_slice.keys():
                n_ax_rows += 1

        trace_str = trace.__str__()
        ea = trace.parameter['Echelle angle']
        pa = trace.parameter['Prism angle']
        tname = trace.parameter['name']
        tag = '$\phi_{prism} = $'
        suptitle = 'fit - zemax displacements for config \n' + "{:s}, {:s}{:6.3f}".format(tname, tag, pa)
        print(suptitle)
        fig, ax_list = Plot.set_plot_area(nrows=n_ax_rows, ncols=1, sharey=False, sharex = True,
                                          xlabel="x / mm", ylabel="y / mm")
        plt.suptitle(suptitle, x=0.13, y=.96, ha='left', fontsize='small')     # x=0., y=0.,
        ax_row = 0
        for slice_no in mfp_projections.keys():
            mfp_slice = mfp_projections[slice_no]
            for spifu_no in mfp_slice:
                mfp_spifu = mfp_slice[spifu_no]
                ax = ax_list[ax_row, 0]
                x_ref, y_ref = mfp_spifu['zemax']['mfp_x'], mfp_spifu['zemax']['mfp_y']
                ax.plot(x_ref, y_ref, color='black', clip_on=True,
                        fillstyle='full', marker='.', ms=5., linestyle='None')
                colours = ['blue', 'red']
                i = 0
                scale = .05
                for key in mfp_spifu:
                    if key == 'zemax':
                        continue
                    mfp = mfp_spifu[key]
                    x, y = mfp['mfp_x'], mfp['mfp_y']
                    u, v = x - x_ref, y - y_ref
                    q = ax.quiver(x_ref, y_ref, u, v,
                                  angles='xy', scale_units='xy', scale=scale,
                                  width=0.0015, color=colours[i])
                    i += 1
                if ax_row == 0:
                    ax.quiverkey(q, X=0.9, Y=1.15, U=0.18, label='10 pixels', labelpos='W', color='black')
                ax_row += 1
        Plot.show()
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
    def dispersion(traces, optical_configuration):
        ax_list = Plot.set_plot_area('Wavelength coverage',
                                     xlabel="Wavelength / micron",
                                     ylabel="Prism angle + 0.02 x Echelle angle ")
        ax = ax_list[0, 0]
        ccs4_colours = mcolors.CSS4_COLORS
        if optical_configuration == Globals.extended:
            ccs4_colours = mcolors.TABLEAU_COLORS

        colours = sorted(ccs4_colours,
                         key=lambda c: tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(c))),
                         reverse=True)
        colour_iterator = iter(colours)
        config_colour = {}

        for trace in traces[0:-1]:
            ech_angle, prism_angle = trace.parameter['Echelle angle'], trace.parameter['Prism angle']
            tag = "{:5.2f}{:5.2f}".format(ech_angle, prism_angle)
            config_colour[tag] = next(colour_iterator, 'black')
            for tf in trace.slice:
                config, matrices, offset_corrections, rays, wave_bounds = tf
                label, slice_no, spifu_no = config
                waves, _, _, det_x, det_y, _, _ = rays
                dw_dlmspix = (waves[1:] - waves[:-1]) / (det_x[1:] - det_x[:-1])
                x = waves[1:]
                y = dw_dlmspix

                # x_label, y_label = x[0], y[0]

                tag = "{:5.2f}{:5.2f}".format(ech_angle, prism_angle)
                colour = config_colour[tag]
                ax.plot(x, y, color=colour, clip_on=True,
                        fillstyle='none', marker='.', mew=1.0, ms=3, linestyle='None')

                label = "{:d}".format(int(label))
                if spifu_no != -1:
                    label += "/{:d}".format(int(spifu_no))
        Plot.show()

    @staticmethod
    def _cycle_ijk(ijk, dir):
        for v in range(0, 3):
            ijk[v] += dir
            if ijk[v] < 0:
                ijk[v] = 2
            if ijk[v] > 2:
                ijk[v] = 0
        return

    @staticmethod
    def _make_colour_list(n_shades_ramp=1, sat=1.):
        """ Make a list of colours by tracking around the colour triangle. 
        """
        n_triangle_edges = 3
        n_colours_ramp = n_shades_ramp * n_triangle_edges
        n_ramps = 2 * n_triangle_edges
        dcol = n_colours_ramp - 1
        drgb = sat / dcol                       # Delta colour
        up = np.arange(0., sat, drgb)
        down = np.arange(sat, 0., -drgb)
        n_colours = n_ramps * dcol
        rgbs = np.zeros((n_colours, 3))         # Array to hold rgb values
        ijk = [0, 1, 2]
        c = 0
        while c < n_colours:
            [i, j, k] = ijk
            rgbs[c:c+dcol, i] = sat
            rgbs[c:c+dcol, j] = up
            rgbs[c:c+dcol, k] = 0.
            c += dcol
            rgbs[c:c+dcol, i] = down
            rgbs[c:c+dcol, j] = sat
            rgbs[c:c+dcol, k] = 0.
            c += dcol
            Plot._cycle_ijk(ijk, 1)

        colours = tuple(rgbs)
        return colours

    @staticmethod
    def series(plot_type, traces, model_config, colour_by='config'):
        _, opticon, _, _, _, _ = model_config
        titles = {'coverage': ('Wavelength coverage', r'$\theta_{prism}$ + 0.1 $\theta_{echelle}$ + det(y) / metre'),
                  'dispersion': ('Dispersion [nm / column]', 'Dispersion [nm / pixel]'),
                  'nm_det': ('Instantaneous wavelength coverage', 'Mosaic coverage [nm]')}
        title, ylabel = titles[plot_type]
        fig, ax_list = Plot.set_plot_area(xlabel=r'Wavelength / $\mu$m',
                                          ylabel=ylabel)
        fig.suptitle(title)
        ax = ax_list[0, 0]
        if colour_by == 'slice':
            slice_rgb = Plot.make_rgb_gradient(np.arange(28))
        if colour_by == 'config':
            slice_rgb = Plot.make_rgb_gradient(np.arange(108))
        n_traces = len(traces)
        for i, trace in enumerate(traces):
            ech_angle, prism_angle = trace.lms_config['ech_ang'], trace.lms_config['pri_ang']
            n_slices, n_spifus = len(trace.unique_slices), len(trace.unique_spifu_slices)   # Multiple slices per trace
            colour = None
            perimeter_upper, perimeter_lower = None, None
            for transform in trace.transforms:
                config = transform.configuration
                keys = transform.slice_specific_kws
                slice_no = config['slice_no']
                spifu_no = config['spifu_no']
                # cfg_id = config['cfg_id']
                cfg = {}
                for key in keys:
                    cfg[key] = config[key]
                cfg['opticon'] = opticon
                waves = trace.get_series('wavelength', **cfg)
                det_x = trace.get_series('det_x', **cfg)
                det_y = trace.get_series('det_y', **cfg)
                if colour_by == 'slice_wave':
                    rgb = Plot.make_rgb_gradient(waves)
                if colour_by == 'slice':
                    colour = slice_rgb[slice_no - 1]
                if colour_by == 'config':
                    idx = n_traces - i
                    colour = slice_rgb[idx]

                x, y = None, None
                if plot_type == 'coverage':
                    x = waves
                    y = prism_angle + 0.1 * ech_angle + 0.004 * spifu_no + 0.001 * det_y
                nm_micron = 1000.0
                if plot_type == 'nm_det':
                    x = waves
                    dw = (trace.wmax - trace.wmin) * nm_micron
                    y = np.full(waves.shape, dw)
                if plot_type == 'dispersion':
                    mm_pix = 0.018
                    dw_dlmspix = -nm_micron * mm_pix * (waves[1:] - waves[:-1]) / (det_x[1:] - det_x[:-1])  # nm / pix
                    x = waves[1:]
                    y = dw_dlmspix

                tag = "{:5.2f}{:5.2f}".format(ech_angle, prism_angle)
                # colour = config_colour[tag]
                n_pts = len(x)

                if colour_by == 'slice_wave':
                    for i in range(0, n_pts):
                        ax.plot(x[i], y[i], color=rgb[i, :], clip_on=True,
                                fillstyle='none', marker='.', mew=1., ms=1, ls='None')
                else:
                    ax.plot(x, y, color=colour, clip_on=True,
                            fillstyle='none', marker='.', mew=1., ms=1, ls='None')

                # Plot perimeter of dot pattern
                unique_waves = np.unique(waves)
                n_waves = len(unique_waves)
                if perimeter_upper is None:
                    perimeter_shape = n_spifus, n_waves, 2
                    perimeter_upper, perimeter_lower = np.zeros(perimeter_shape), np.zeros(perimeter_shape)
                unique_slices = trace.unique_slices
                unique_spifus = trace.unique_spifu_slices
                spifu_idx = np.argwhere(spifu_no == unique_spifus)[0][0]
                is_slice_lower = slice_no == unique_slices[0]
                is_slice_upper = slice_no == unique_slices[-1]
                for wave_idx, uw in enumerate(unique_waves):
                    indices = np.argwhere(x == uw)
                    yp_unsort = y[indices]
                    yp = np.sort(yp_unsort)
                    if is_slice_lower:
                        perimeter_lower[spifu_idx, wave_idx] = [uw, yp[-1][0]]
                    if is_slice_upper:
                        perimeter_upper[spifu_idx, wave_idx] = [uw, yp[0][0]]

            for spifu_idx, spifu_no in enumerate(trace.unique_spifu_slices):
                xyl = perimeter_lower[spifu_idx]
                xyu = perimeter_upper[spifu_idx]
                xyl = np.flip(xyl, axis=0)
                xy = np.concatenate((xyu, xyl, xyu[0:1]))
                plt.fill(xy[:, 0], xy[:, 1], color='pink')
                ax.plot(xy[:, 0], xy[:, 1], color='black', linestyle='solid', lw=0.5)

        Plot.show()
        return

    @staticmethod
    def plot_points(ax, x, y, **kwargs):
        """ Plot an array of points in the open plot region.
        """
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
                ax.plot(x[i], y[i], color=rgb[i, :], clip_on=True,
                        fillstyle=fs, marker=mk, mew=mew, ms=ms)
        return

    @staticmethod
    def show():
        """ Wrapper for matplotlib show function. """
        import matplotlib.pyplot as plt
        plt.show()
