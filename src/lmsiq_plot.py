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
        images, obs_dict = observations
        im_map = None
        png_path = kwargs.get('png_path', None)
        nrowcol = kwargs.get('nrowcol', (1, 1))
        shrink = kwargs.get('shrink', 1.0)
        title = kwargs.get('title', '')
        pane_titles = kwargs.get('pane_titles', None)
        colourbar = kwargs.get('colourbar', True)

        n_rows, n_cols = nrowcol
        difference = kwargs.get('difference', False)
        ref_img = images[0]
        ny, nx = ref_img.shape
        # Only plot the central part of the image
        box_rad = int(0.5 * shrink * ny)
        r_cen, c_cen = int(ny / 2), int(nx / 2)
        r1, r2, c1, c2 = r_cen - box_rad, r_cen + box_rad + 1, c_cen - box_rad, c_cen + box_rad + 1

        if difference:
            img_diffs = []
            for img in images:
                img_diff = img - ref_img
                img_diffs.append(img_diff)
            observations = img_diffs, obs_dict
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
            images, obs_dict = observations
            for img in images:
                vmin = vmin if vmin < np.amin(img) else np.amin(img)
                vmax = vmax if vmax > np.amax(img) else np.amax(img)

        file_names = obs_dict['file_names']
        for i, image in enumerate(images[0:4]):
            ax = ax_list[row, col]
            if pane_titles is not None:
                pane_title = file_names[i] if pane_titles == 'file_id' else pane_titles[pane]
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
    def collage(image_list, obs_dict, **kwargs):
        png_path = kwargs.get('png_path', None)
        # config, slice_label, wave, wave_no, slice_no, ipc_tag = parameters
        # optical_path, date_stamp, n_wavelengths, n_mcruns, zim_locs, _, folder_name, config_label = config
        pane_titles = ['Perfect', 'Design', 'M-C run 0', 'M-C run 1']
        wave = obs_dict['wavelength']
        config_no = obs_dict['config_no']
        slice_no = obs_dict['slice_no']
        fmt = "config={:d}, slice={:d} $\lambda$ = {:5.3f} $\mu$m"
        title = fmt.format(config_no, slice_no, wave)
#        slice_tag = "_slice{:d}".format(slice_no) if slice_no > 0 else "slicer"
        observations = image_list, obs_dict
        Plot.images(observations,
                    nrowcol=(2, 2), shrink=1.0,
                    title=title, pane_titles=pane_titles, do_log=True, png_path=png_path)
        return

    @staticmethod
    def _make_profile_title(axis, wav, plot_label, ipc_tag):
        fmt = r' $\lambda$ =' + "{:7.3f} um, {:s}"
        title = plot_label + '\n' + axis + ' axis,' + fmt.format(wav, ipc_tag)
        return title

    @staticmethod
    def plot_ee(ees_data, obs_dict, axis, plot_label, ipc_tag, **kwargs):
        png_path = kwargs.get('png_path', None)
        plot_all = kwargs.get('plot_all', True)

        key_list = []
#        axis, xlms, y_per, y_des, y_mean, y_rms, y_mcs = ee_data
        wavelength = obs_dict['wavelength']
        title = Plot._make_profile_title(axis, wavelength, plot_label, ipc_tag)

        fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
        ax = ax_list

        xlms = ees_data['xdet']
        x_ref = 3.

        a = np.log10(2. * xlms)
        a_ref = math.log10(2. * x_ref)
        xlim = [a[0], a[-1]]
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
            if plot_all:
                for key in ees_data:
                    if 'MC' in key:
                        y = ees_data[key]
                        ax.plot(xlms, y, color='grey', lw=0.5)

        plot_key_data = {'mean': ('<model>', 0, 'blue'),
                         'per': ('perfect', 1, 'red'),
                         'des': ('design', 2, 'green'),
                         }
        # for plot_key in plot_key_data:
        #     for ee_key in ee_refs:
        #         if plot_key in ee_key:
        #             key_label, y_col, colour = plot_key_data[plot_key]
        #             y = y_mcs[:, y_col]
        #             ee_ref = ee_refs[ee_key]
        #             ax.plot(a, y, color=colour)
        #             ax.plot(a_ref, ee_ref, color=colour, marker='o')
        #             key_list.append((key_label, colour, [ee_ref]))

        key_title = "EE at '{:s}'={:5.2f} pix.".format(ax_xtag, 2 * x_ref)
        Plot.key(ax, axis, key_list, xlim, ylim,
                 fmts=['{:>8.3f}'], col_labels=None, title=key_title)

        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def plot_lsf(lsf_data, obs_dict, axis, dw_pix, plot_label, ipc_tag, line_widths, **kwargs):
        """ Plot line spread functions.  y_all element of lsf_data holds the perfect and esign profiles
        in index 0 and 1,
        :param axis:
        :param obs_dict:
        :param ipc_tag:
        :param line_widths:   FWHM data (det. pixels) for this configuration
        :param plot_label:
        :param lsf_data:
        :param dw_pix:
        :param kwargs:
        :return:
        """
        png_path = kwargs.get('png_path', None)
        plot_all = kwargs.get('plot_all', True)
        xlim_det = kwargs.get('xlim_det', 5.0)

        # axis, x, y_per, y_des, y_mean, y_rms, y_mcs = lsf_data
        wavelength = obs_dict['wavelength']
        title = Plot._make_profile_title(axis, wavelength, plot_label, ipc_tag)

        fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
        ax = ax_list

        x = lsf_data['xdet']
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
            for key in lsf_data:
                if 'MC' in key:
                    y = lsf_data[key]
                    ax.plot(x, y, color='grey', lw=0.5)

        colours = {'perfect': 'red', 'design': 'green', '<mean>': 'blue'}

        for key in lsf_data:
            if key not in colours:
                continue
            colour = colours[key]
            xfwhm, xl, xr = line_widths[key]
            y = lsf_data[key]
            yh = 0.5
            label = "{:s}, {:4.2f}".format(key, xfwhm)
            ax.plot(x, y, color=colour, lw=2.0, label=label)
            Plot._hwarrow(ax, 'right', xl, yh, 0.5, 0.02, colour)
            Plot._hwarrow(ax, 'left', xr, yh, 0.5, 0.02, colour)
        ax.legend()
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
        ax.plot(x, y, color=colour, lw=2.0)
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
    def phase_shift(data_type, data_table, obs_dict, wave, ipc_tag, **kwargs):
        png_path = kwargs.get('png_path', None)

        wave = obs_dict['wavelength']
        slice_no = obs_dict['slice_no']
        spifu_no = obs_dict['spifu_no']

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        xlabel = 'Image Shift [pix.]'
        ylabel = 'Value'
        ax.set_xlabel(xlabel, fontsize=16.0)
        ax.set_ylabel(ylabel, fontsize=16.0)
        fmt = "{:s} variation at {:5.3f} um \n ({:s}, spatial slice={:02d}, spectral slice={:02d})"
        title = fmt.format(data_type, wave, ipc_tag, slice_no, spifu_no)
        ax.set_title(title)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        styles = {'pe': {'color': 'green', 'lw': 1.0, 'mew': 2.0, 'marker': '+'},
                  'de': {'color': 'orange', 'lw': 1.0, 'mew': 2.0, 'marker': '+'},
                  'MC': {'color': 'grey', 'lw': 0.5, 'mew': 1.0, 'marker': 'none'},
                  }

        x = data_table['phase shift']
        keys = list(data_table.keys())
        for key in keys[1:]:
            style_key = key[0:2]
            style = styles[style_key]
            style['ls'] = 'solid'

            y = data_table[key]
            ax.plot(x, y, **style)

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
            ax.plot(det_shift, y, lw=0.5, marker='+', mew=2.0)

        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def rms_vals(waves, rms_vals, **kwargs):
        png_path = kwargs.get('png_path', None)
        title = kwargs.get('title', '')

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        xlabel = 'Wavelength [micron]'
        ylabel = 'RMS fractional change'
        ax.set_xlabel(xlabel, fontsize=16.0)
        ax.set_ylabel(ylabel, fontsize=16.0)
        ax.set_title(title)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for ipc_key in rms_vals:
            obs_table = rms_vals[ipc_key]
            y_table = {'pe': [], 'de': [], 'MC': []}
            for obs_record in obs_table:
#                config_table = obs_table[obs_key]
                mc_sum = 0.
                for obs_key in obs_record:
                    rms_val = obs_record[obs_key]
                    y_key = obs_key[0:2]
                    if y_key == 'MC':
                        mc_sum += rms_val
                    else:
                        y_table[y_key].append(rms_val)
                mc_mean = np.mean(np.array(mc_sum))
                y_table['MC'].append(mc_mean)

            for key in y_table:
                ax.plot(waves, y_table[key], lw=0.5, marker='+', mew=2.0, label=ipc_key + key)
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
            ax.plot(w, y, ls=ls, lw=1.0, **key_words)
            if plot_errors:
                ax.plot(w, y - yerr, ls='dotted', lw=1.0, color=colours[ipc_tag])
                ax.plot(w, y + yerr, ls='dotted', lw=1.0, color=colours[ipc_tag])
            if srp_req:
                w_metis2745 = [2.7, 4.8, 4.8, 5.5]
                y_metis2745 = [100000, 100000, 85000, 85000]
                ax.plot(w_metis2745, y_metis2745, ls='dashed', lw=2.0, color='red')
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
        ax.plot(xp, yp, **kwargs)
        ax.plot(xl, yl, **kwargs)
        return
