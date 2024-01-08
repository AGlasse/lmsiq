import math
import numpy as np
import matplotlib.rcsetup as rcs
import matplotlib.pyplot as plt


class Plot:

    def __init__(self):
        plt.rcParams['backend'] = 'AGG'
        return

    @staticmethod
    def set_plot_area(**kwargs):
        figsize = kwargs.get('figsize', [12, 9])
        title = kwargs.get('title', 'title')
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

        for i in range(0, nrows):
            for j in range(0, ncols):
                ax = ax_list[i, j]
                ax.set_aspect(aspect)       # Set equal axes
                if xlim is not None:
                    ax.set_xlim(xlim)
                if ylim is not None:
                    ax.set_ylim(ylim)
                if i == nrows-1 and j == 0:
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
        if remplots is not None:
            rps = np.atleast_2d(remplots)
            for i in range(0, len(rps)):
                ax_list[rps[i, 0], rps[i, 1]].remove()
        return fig, ax_list

    @staticmethod
    def show():
        """ Wrapper for matplotlib show function. """
        plt.show()

    @staticmethod
    def images(observations, **kwargs):
        """ Plot images from the first four observations (perfect, design and as many additional individual
        models as will fit in the grid.
        """
        im_map = None
        png_path = kwargs.get('png_path', None)
        nrowcol = kwargs.get('nrowcol', (1, 1))
        shrink = kwargs.get('shrink', 1.0)
        title = kwargs.get('title', '')
        pane_titles = kwargs.get('pane_titles', None)
        colourbar = kwargs.get('colourbar', True)

        n_rows, n_cols = nrowcol
        difference = kwargs.get('difference', False)
        obs_diffs = []
        ref_img, _ = observations[0]
        ny, nx = ref_img.shape
        # Only plot the central part of the image
        box_rad = int(0.5 * shrink * ny)
        r_cen, c_cen = int(ny / 2), int(nx / 2)
        r1, r2, c1, c2 = r_cen - box_rad, r_cen + box_rad + 1, c_cen - box_rad, c_cen + box_rad + 1

        if difference:
            for obs in observations:
                img, par = obs
                img_diff = obs[0] - ref_img
                obs_diff = img_diff, par
                obs_diffs.append(obs_diff)
            observations = obs_diffs
        xlim, ylim = [c1-1., c2], [r1-1., r2]
        fig, ax_list = Plot.set_plot_area(**kwargs, nrows=n_rows, ncols=n_cols,
                                          xlim=xlim, ylim=ylim)
        ax_list = np.atleast_2d(ax_list)

        fig.suptitle(title)
        # Initialise variables and get plot limits
        pane, row, col = 0, 0, 0
        do_log = kwargs.get('do_log', True)
        do_half = False
        vmin, vmax = kwargs.get('vmin', None), kwargs.get('vmax', None)
        if vmin is None:
            vmin, vmax = np.finfo('float').max, np.finfo('float').min
            for obs in observations:
                img, _ = obs
                vmin = vmin if vmin < np.amin(img) else np.amin(img)
                vmax = vmax if vmax > np.amax(img) else np.amax(img)

        for observation in observations:
            image, params = observation
            file_id, _ = params
            ax = ax_list[row, col]
            if pane_titles is not None:
                pane_title = file_id if pane_titles == 'file_id' else pane_titles[pane]
                ax.set_title(pane_title)

            if do_log:
                lvmax = math.log10(vmax)
                vmin = vmax / 1000.0
                lvmin = math.log10(vmin)
                clipped_image = np.where(image < vmin, vmin, image)
                log_image = np.log10(clipped_image)
                im_map = ax.imshow(log_image[r1:r2, c1:c2],
                                   extent=(c1-0.5, c2-0.5, r1-0.5, r2-0.5),
                                   vmin=lvmin, vmax=lvmax)
            else:
                if do_half:
                    vmax = np.amax(image)
                    vmin = vmax / 2.0
                im_map = ax.imshow(image[r1:r2, c1:c2],
                                   extent=(c1-1.5, c2+1.5, r1-1.5, r2+1.5),
                                   vmin=vmin, vmax=vmax)
            pane += 1
            col += 1
            if col == n_cols:
                col = 0
                row += 1
        if colourbar:
            bar_label = 'Log10(Signal)' if do_log else 'Signal'
            plt.colorbar(mappable=im_map, ax=ax_list, label=bar_label, shrink=0.75)
        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def collage(obs_list, parameters, **kwargs):
        png_path = kwargs.get('png_path', None)
        config, slice_label, wave, wave_no, slice_no, ipc_tag = parameters
        optical_path, date_stamp, n_wavelengths, n_mcruns, zim_locs, folder_name, config_label = config
        pane_titles = ['Perfect', 'Design', 'M-C run 0', 'M-C run 1']
        fmt = "{:s}, {:s}{:s}{:s}{:s} $\lambda$ = {:5.3f} $\mu$m"
        title = fmt.format(optical_path, date_stamp, '', config_label, slice_label, wave)
        slice_tag = "_slice{:d}".format(slice_no) if slice_no > 0 else "slicer"

        Plot.images(obs_list,
                    nrowcol=(2, 2), shrink=1.0,
                    title=title, pane_titles=pane_titles, do_log=True, png_path=png_path)
        return

    @staticmethod
    def _make_profile_title(axis, wav, plot_label, ipc_tag):
        fmt = r' $\lambda$ =' + "{:7.3f} um, {:s}"
        title = plot_label + '\n' + axis + ' axis,' + fmt.format(wav, ipc_tag)
        return title

    @staticmethod
    def plot_ee(ee_data, wav, x_ref, ee_refs, plot_label, ipc_tag, **kwargs):
        png_path = kwargs.get('png_path', None)
        plot_all = kwargs.get('plot_all', True)

        key_list = []
        axis, xlms, y_per, y_des, y_mean, y_rms, y_mcs = ee_data
        title = Plot._make_profile_title(axis, wav, plot_label, ipc_tag)

        fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
        ax = ax_list
        n_points, n_files = y_mcs.shape

        colours = ['grey'] * n_files
        colours[0] = 'red'
        colours[1] = 'black'
        a = np.log10(2. * xlms)
        a_ref = math.log10(2. * x_ref)
        xlim = [a[0], a[n_points-1]]
        ylim = [0.0, 1.05]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        xtick_lin_vals = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
        xtick_vals = np.log10(xtick_lin_vals)
        ax.set_xticks(xtick_vals)
        ax.set_xticklabels(xtick_lin_vals)
        ax_xlabel = 'width'
        ax_xtag = 'w'

        if axis == 'spatial':
            ax_xlabel = 'height'
            ax_xtag = 'h'
        ax_ylabel = "Enslitted energy fraction 'EE({:s})'".format(ax_xtag)
        if axis == 'radial':
            ax_ylabel = "Encircled energy fraction 'EE(r)'"
            ax_xlabel, ax_xtag = 'radius', 'r'
        ax.set_ylabel(ax_ylabel, fontsize=16.0)
        ax.set_xlabel("Aperture {:s} '{:s}' (det. pixels)".format(ax_xlabel, ax_xtag), fontsize=16.0)

        ax.set_title(title, fontsize=16.0)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        ax.xaxis.grid()
        ax.yaxis.grid()

        if plot_all:
            for j in range(0, n_files):
                y = y_mcs[:, j]
                ax.plot_focal_planes(a, y, color='grey', lw=0.5)

        plot_key_data = {'mean': ('<model>', 0, 'blue'),
                         'per': ('perfect', 1, 'red'),
                         'des': ('design', 2, 'green'),
                         }
        for plot_key in plot_key_data:
            for ee_key in ee_refs:
                if plot_key in ee_key:
                    key_label, y_col, colour = plot_key_data[plot_key]
                    y = y_mcs[:, y_col]
                    ee_ref = ee_refs[ee_key]
                    ax.plot_focal_planes(a, y, color=colour)
                    ax.plot_focal_planes(a_ref, ee_ref, color=colour, marker='o')
                    key_list.append((key_label, colour, [ee_ref]))

        key_title = "EE at '{:s}'={:5.2f} pix.".format(ax_xtag, 2 * x_ref)
        Plot.key(ax, axis, key_list, xlim, ylim,
                 fmts=['{:>8.3f}'], col_labels=None, title=key_title)

        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def plot_lsf(lsf_data, wav, dw_pix, plot_label, ipc_tag, key_line_widths, **kwargs):
        """ Plot line spread functions.  y_all element of lsf_data holds the perfect and esign profiles
        in index 0 and 1,
        :param ipc_tag:
        :param key_line_widths:   FWHM data (det. pixels) for this configuration
        :param plot_label:
        :param lsf_data:
        :param wav:
        :param dw_pix:
        :param kwargs:
        :return:
        """
        png_path = kwargs.get('png_path', None)
        plot_all = kwargs.get('plot_all', True)
        xlim_det = kwargs.get('xlim_det', 5.0)

        axis, x, y_per, y_des, y_mean, y_rms, y_mcs = lsf_data
        title = Plot._make_profile_title(axis, wav, plot_label, ipc_tag)

        fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
        ax = ax_list

        n_points, n_mcs = y_mcs.shape

        xlim = [-xlim_det, xlim_det]
        ylim = [0.0, 1.05]
        xtick_interval = 1.0 if xlim[1] < 5.0 else 2.0
        xtick_lin_vals = np.arange(xlim[0], xlim[1] + 1.0, xtick_interval)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(xtick_lin_vals)
        ax.set_xticklabels(xtick_lin_vals)
        ax_ylabel = 'Spectral'
        ax_xtag = 'x'
        if axis == 'spatial':
            ax_ylabel = 'Spatial'
            ax_xtag = 'y'

        axis_font = 16.0
        title_font = 14.0
        ax.set_ylabel("{:s} Profile 'f({:s})'".format(ax_ylabel, ax_xtag), fontsize=axis_font)
        ax.set_xlabel("'{:s}' (det. pixels)".format(ax_xtag), fontsize=axis_font)
        ax.set_title(title, fontsize=title_font)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(axis_font)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(axis_font)
        if plot_all:
            for j in range(0, n_mcs):
                y = y_mcs[:, j]
                ax.plot_focal_planes(x, y, color='grey', lw=0.5)

        key_list = []
        key_profiles = [y_per, y_des, y_mean]
        colours = {'perfect': 'red', 'design': 'green', '<mean>': 'blue'}
        for row, key in enumerate(colours):
            colour = colours[key]
            xl, xr = key_line_widths[row, 1], key_line_widths[row, 2]
            y = key_profiles[row]
            Plot._lsf_line(ax, axis, key, wav, dw_pix, x, y, xl, xr, colour, key_list)

        Plot.key(ax, axis, key_list, xlim, ylim,
                 fmts=["{:>8.2f}", "{:>8.0f}"])
        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def cube_strehls(waves, strehl_list, title, **kwargs):
        png_path = kwargs.get('png_path', None)

        axis_font = 16.0
        title_font = 14.0

        fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
        ax = ax_list

        xlim = [2.6, 5.4]
        ylim = [0.6, 1.05]

        ax.set_title(title)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel('Wavelength [$\mu$m]', fontsize=axis_font)
        ax.set_ylabel('Strehl ratio', fontsize=axis_font)

        colours = {'ipc_off': 'black', 'ipc_on': 'green'}
        for strehls, ipc_tag in strehl_list:
            y, yerr = strehls[:, 2], strehls[:, 3]
            ax.errorbar(waves, y,
                        yerr=yerr, marker='o', mew=2.0, label=ipc_tag,
                        color=colours[ipc_tag])
        plt.legend()

        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def _lsf_line(ax, axis, key, wav, dw_pix, x, y, xl, xr, colour, key_list):
        yh = 0.5
        ax.plot_focal_planes(x, y, color=colour, lw=2.0)
        Plot._hwarrow(ax, 'right', xl, yh, 0.5, 0.02, colour)
        Plot._hwarrow(ax, 'left', xr, yh, 0.5, 0.02, colour)
        xfwhm = xr - xl
        srp = wav / (dw_pix * (xr - xl))
        vals = [xfwhm] if axis == 'spatial' else [xfwhm, srp]
        key_list.append((key, colour, vals))
        return

    @staticmethod
    def phase_summary(stats_list, **kwargs):
        png_path = kwargs.get('png_path', None)
        sigma = kwargs.get('sigma', 1.)

        title = "Phase shift rate error - (Error bars scaled by {:5.2f})".format(sigma)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.set_xlabel('Wavelength [$\mu$m]', fontsize=16.0)
        ax.set_ylabel('RMS (LSF shift / PSF movement)', fontsize=16.0)
        ax.set_title(title, fontsize=18.0)
        for config_tag, stats in stats_list:
            waves, phase_err, phase_err_std = stats[:, 0], stats[:, 1], stats[:, 2]
            yerr = phase_err_std * sigma
            label = config_tag[1:]
            ax.errorbar(waves, phase_err, yerr=yerr,  marker='+', mew=2.0, label=label)
        plt.legend()
        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def phase_shift(data_type, phase_data, wave, **kwargs):
        png_path = kwargs.get('png_path', None)
        mc_only = kwargs.get('mc_only', False)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        xlabel = 'Image Shift [pix.]'
        ylabel = 'Value'
        ax.set_xlabel(xlabel, fontsize=16.0)
        ax.set_ylabel(ylabel, fontsize=16.0)
        title = "{:s} variation at {:5.3f} micron".format(data_type, wave)
        ax.set_title(title)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)

        det_shift = phase_data[:, 0]
        _, n_cols = phase_data.shape
        start_col = 3 if mc_only else 1

        for col in range(start_col, n_cols):
            y = phase_data[:, col]
            y_mean = np.mean(y)
#            y_mean = 0.
            y -= y_mean
            ax.plot_focal_planes(det_shift, y, lw=0.5, marker='+', mew=2.0)

        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def photometry_scatter(photometry, wave, **kwargs):
        png_path = kwargs.get('png_path', None)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        xlabel = 'Image Shift [pix.]'
        ylabel = 'Fractional change in photometry'
        ax.set_xlabel(xlabel, fontsize=16.0)
        ax.set_ylabel(ylabel, fontsize=16.0)
        title = "Photometric variation at {:5.3f} micron".format(wave)
        ax.set_title(title)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)

        det_shift = photometry[:, 0]
        _, n_cols = photometry.shape
        for col in range(1, n_cols):
            y = photometry[:, col]
            y_mean = np.mean(y)
            y -= y_mean
            ax.plot_focal_planes(det_shift, y, lw=0.5, marker='+', mew=2.0)

        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def rms_vals(waves, phot_rms_dict, **kwargs):
        png_path = kwargs.get('png_path', None)
        sigma = kwargs.get('sigma', 1.)
        title = kwargs.get('title', '')

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        xlabel = 'Image Shift [pix.]'
        ylabel = 'Fractional change'
        ax.set_xlabel(xlabel, fontsize=16.0)
        ax.set_ylabel(ylabel, fontsize=16.0)
        ax.set_title(title)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)

        for i, phot_key in enumerate(phot_rms_dict):
            phot_rms = phot_rms_dict[phot_key]
            ax.plot_focal_planes(waves, phot_rms, lw=0.5, marker='+', mew=2.0, label=phot_key)
        plt.legend()

        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def profile(profile, profile_data_list, **kwargs):
        srp_req = kwargs.get('srp_req', False)
        png_path = kwargs.get('png_path', None)
        plot_errors = kwargs.get('plot_errors', False)
        ls = kwargs.get('ls', 'none')
        xlabel = 'Wavelength [$\mu$m]'
        ylabel = kwargs.get('ylabel', 'Value')
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.set_xlabel(xlabel, fontsize=16.0)
        ax.set_ylabel(ylabel, fontsize=16.0)
        ax.set_xlim([2.5, 5.6])
        ax.set_title(profile, fontsize=14.0)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)

        colours = {'ipc_off': 'black', 'ipc_on': 'green'}
        for profile_data in profile_data_list:
            plot_id, config, profile_dict, profiles, ipc_tag = profile_data
            config_number, _, ls, marker = plot_id
            fillstyle = 'none'
            w_idx = profile_dict['wave']
            w = profiles[:, w_idx]
            label = ipc_tag
            idx = profile_dict[profile]
            y = profiles[:, idx]
            if plot_errors:
                yerr = profiles[:, idx + 1]
            key_words = {'marker': marker, 'ms': 10., 'mew': 2.0, 'fillstyle': fillstyle,
                         'label': label, 'color': colours[ipc_tag]}
            ax.plot_focal_planes(w, y, ls=ls, lw=1.0, **key_words)
            if plot_errors:
                ax.plot_focal_planes(w, y - yerr, ls='dotted', lw=1.0, color=colours[ipc_tag])
                ax.plot_focal_planes(w, y + yerr, ls='dotted', lw=1.0, color=colours[ipc_tag])
            if srp_req:
                w_metis2745 = [2.7, 4.8, 4.8, 5.5]
                y_metis2745 = [100000, 100000, 85000, 85000]
                ax.plot_focal_planes(w_metis2745, y_metis2745, ls='dashed', lw=2.0, color='red')
        plt.legend()
        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def key(ax, axis, key_list, xlim, ylim, **kwargs):
        fmts = kwargs.get('fmts', ['{:>6.2f}', '{:>6.2f}'])
        col_labels = kwargs.get('col_labels', ['FWHM/pix.', 'SRP'])
        title = kwargs.get('title', None)

        xr, yr = xlim[1] - xlim[0], ylim[1] - ylim[0]
        x1 = xlim[0] + 0.02 * xr
        x2 = x1 + 0.15 * xr
        x3 = x1 + 0.30 * xr
        x_text = [x2, x3]
        y = ylim[1] - 0.04 * yr
        if title is not None:
            ax.text(x1, y, title, color='black', fontsize=14.0)
        if col_labels is not None:
            y -= 0.05
            ax.text(x2, y, col_labels[0], color='black', fontsize=14.0)
            if axis == 'spectral':
                ax.text(x3, y, col_labels[1], color='black', fontsize=14.0)
        for key in key_list:
            y -= 0.05 * yr
            text, colour, vals = key
            text_params = {'color': colour, 'fontsize': 14.0}
            lab = "{:<8s}".format(text)
            ax.text(x1, y, lab, **text_params)
            for i, val in enumerate(vals):
                num = fmts[i].format(val)
                x = x_text[i]
                ax.text(x, y, num, **text_params)
        return

    @staticmethod
    def _hwarrow(ax, direction, x, y, width, height, colour):
        """ Draw an arrow to mark the profile half-width.
        """
        width = -width if direction == 'right' else width
        dx, dy = 0.3 * width, height
        xp, xl = [x + dx, x, x + dx], [x, x + width]
        yp, yl = [y + dy, y, y - dy], [y, y]
        kwargs = {'color': colour, 'lw': 1.5}
        ax.plot_focal_planes(xp, yp, **kwargs)
        ax.plot_focal_planes(xl, yl, **kwargs)
        return
