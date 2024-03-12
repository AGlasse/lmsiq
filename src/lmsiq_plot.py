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
    def plot_cube_profile(type_key, data, name, ds_dict, axis, **kwargs):
        xlog = kwargs.get('xlog', False)
        png_path = kwargs.get('png_path', None)
        plot_all = kwargs.get('plot_all', True)

        wavelength = ds_dict['wavelength']
        data_pars = {'ee': {'title_lead': 'Enslitted energy',
                            'xlabels': {'spectral': 'aperture height',
                                        'spatial': 'aperture width',
                                        'across-slice': 'aperture width',
                                        'along-slice': 'aperture height'},
                            'ylabels': {'spectral': 'EE(x)',
                                        'spatial': 'EE(y)',
                                        'across-slice': 'EE(x)',
                                        'along-slice': 'EE(y)'},
                            },
                     'lsf': {'title_lead': 'Line spread function',
                             'xlabels': {'spectral': 'spectral',
                                         'spatial': 'spatial',
                                         'across-slice': 'across slice',
                                         'along-slice': 'along slice'},
                             'ylabels': {'spectral': 'signal',
                                         'spatial': 'signal',
                                         'across-slice': 'signal',
                                         'along-slice': 'signal'},
                             }
                     }
        data_par = data_pars[type_key]

        title_lead = data_par['title_lead']
        title = "{:s}\n{:s}, $\lambda$= {:5.3f} $\mu$m".format(title_lead, name, wavelength)

        fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
        ax = ax_list

        x = data['xvals']
        ys = data['yvals']
        n_pts, n_profiles = ys.shape
        ax_xlabel = data_par['xlabels'][axis]
        ax_xtag = 'w'

        ax_ylabel = data_par['ylabels'][axis]
        if axis == 'radial':
            ax_ylabel = "Encircled energy fraction 'EE(r)'"
            ax_xlabel, ax_xtag = 'radius', 'r'
        ax.set_ylabel(ax_ylabel, fontsize=16.0)
        ax.set_xlabel("{:s} '{:s}' (det. pixels)".format(ax_xlabel, ax_xtag), fontsize=16.0)

        ax.set_title(title, fontsize=16.0)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        ax.xaxis.grid()
        ax.yaxis.grid()
        if xlog:
            ax.set_xscale('log')

        plot_key_data = {0: ('perfect', 'red'),
                         1: ('design', 'green'),
                         2: ('<model>', 'blue')
                         }
        for prof_idx in range(0, n_profiles):
            if prof_idx in plot_key_data:
                label, colour = plot_key_data[prof_idx]
                if type_key == 'lsf':
                    xfwhm, xl, xr = data['fwhm_lin'][prof_idx], data['xl'][prof_idx], data['xr'][prof_idx]
                    label += " {:5.2f}".format(xfwhm)
                y = ys[:, prof_idx]
                ax.plot(x, y, color=colour, lw=1.5, label=label)
            else:
                if plot_all:
                    y = ys[:, prof_idx]
                    ax.plot(x, y, color='grey', lw=0.5)
        ax.legend()

        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def strehls(cube_series, **kwargs):
        data_tag = 'strehl'                               # Tag identifying fwhm data in cube_series
        png_path = kwargs.get('png_path', None)
        select = kwargs.get('select', {})
        ordinate = kwargs.get('ordinate', 'fwhm')
        ordinate_unit = kwargs.get('ordinate_unit', 'pixels')
        nvals = len(cube_series['ipc_on'])
        sel_indices = np.full(nvals, True)
        for key in select:
            val = select[key]
            series = np.array(cube_series[key])
            sel_indices = np.logical_and(sel_indices, series == val)

        fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
        ax = ax_list
        xlabel = 'Wavelength / $\mu$m'
        ylabel = ordinate + ' ' + ordinate_unit
        ax.set_ylabel(ylabel, fontsize=16.0)
        ax.set_xlabel(xlabel, fontsize=16.0)

        w_unsorted = np.array(cube_series['wavelength'])[sel_indices]
        sorted_indices = np.argsort(w_unsorted)
        plot_series = {}
        for key in cube_series:
            cube_array = np.array(cube_series[key])
            plot_array = cube_array[sel_indices]
            plot_series[key] = plot_array[sorted_indices]

        y_table = {}
        y_mc_list = []
        x = plot_series['wavelength']
        for key in cube_series:
            if data_tag in key:
                y = plot_series[key]
                if ordinate == 'srp':
                    dw_dlmspix = plot_series['dw_dlmspix']
                    y = x * 1000. / (y * dw_dlmspix)
                if 'MC' in key:                             # Find min and max of MC FWHM values
                    y_mc_list.append(y)
                else:
                    y_table[key] = y
        is_mc_data = len(y_mc_list) > 0
        if is_mc_data:
            y_mcs = np.array(y_mc_list)
            y_mc_lo, y_mc_hi = np.amin(y_mcs, axis=0), np.amax(y_mcs, axis=0)
            y_mc_mean = np.mean(y_mcs, axis=0)
            y_table['mc_mean'] = y_mc_mean

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        ax.xaxis.grid()
        ax.yaxis.grid()
        if is_mc_data:
            ax.fill_between(x, y_mc_lo, y_mc_hi, color='skyblue')
        plot_keys = {'mc_mean': ('<model>', 'blue'),
                     'perfect': ('perfect', 'red'),
                     'design': ('design', 'green')}
        handles = []
        for key in y_table:
            y = y_table[key]
            for plot_key in plot_keys:
                if plot_key in key:
                    label, colour = plot_keys[plot_key]
                    handle, = ax.plot(x, y, marker='>', mew=2.0, label=label,
                                      ls='none', fillstyle='none', color=colour)
                    handles.append(handle)

        ax.legend(handles=handles)

        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()

    @staticmethod
    def wav_series(cube_series, **kwargs):
        png_path = kwargs.get('png_path', None)
        select = kwargs.get('select', {})
        ordinate = kwargs.get('ordinate', 'fwhms')
        ordinate_unit = kwargs.get('ordinate_unit', 'pixels')
        plot_all = kwargs.get('plot_all', True)
        nvals = len(cube_series['ipc_on'])
        sel_indices = np.full(nvals, True)
        for key in select:
            val = select[key]
            series = np.array(cube_series[key])
            sel_indices = np.logical_and(sel_indices, series == val)

        fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
        ax = ax_list
        xlabel = 'Wavelength / $\mu$m'
        ylabel = ordinate + ' ' + ordinate_unit
        ax.set_ylabel(ylabel, fontsize=16.0)
        ax.set_xlabel(xlabel, fontsize=16.0)

        w_unsorted = np.array(cube_series['wavelength'])[sel_indices]
        sorted_indices = np.argsort(w_unsorted)
        plot_series = {}
        for key in cube_series:
            cube_array = np.array(cube_series[key])
            plot_array = cube_array[sel_indices]
            plot_series[key] = plot_array[sorted_indices]

        y_table = {}
        x = plot_series['wavelength']

        yvals = plot_series['strehls'] if ordinate == 'strehls' else plot_series['fwhms']
        n_pts, n_profiles = yvals.shape
        if ordinate == 'srps':
            dw_dlmspix = plot_series['dw_dlmspix']
            for i in range(0, n_pts):
                for j in range(0, n_profiles):
                    dw = yvals[i, j] * dw_dlmspix[i]
                    yvals[i, j] = x[i] * 1000. / dw
        y_mcs = yvals[:, 2:]
        y_mc_lo, y_mc_hi = np.amin(y_mcs, axis=1), np.amax(y_mcs, axis=1)
        y_mc_mean = np.mean(y_mcs, axis=0)
        y_table['mc_mean'] = y_mc_mean

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        ax.xaxis.grid()
        ax.yaxis.grid()
        ax.fill_between(x, y_mc_lo, y_mc_hi, color='skyblue')
        plot_key_data = {0: ('perfect', 'red'),
                         1: ('design', 'green'),
                         2: ('<model>', 'blue')}
        handles = []
        for prof_idx in range(0, n_profiles):
            if prof_idx in plot_key_data:
                label, colour = plot_key_data[prof_idx]
                y = yvals[:, prof_idx]
                handle, = ax.plot(x, y, color=colour, lw=1.5, label=label)
                handles.append(handle)
            else:
                if plot_all:
                    y = yvals[:, prof_idx]
                    ax.plot(x, y, color='grey', lw=0.5)

        ax.legend(handles=handles)

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
