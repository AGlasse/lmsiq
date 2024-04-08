import math
import numpy as np
import matplotlib.pyplot as plt


class Plot:

    def __init__(self):
        plt.rcParams['backend'] = 'AGG'
        return

    @staticmethod
    def set_plot_area(**kwargs):
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

        for i in range(0, nrows):
            for j in range(0, ncols):
                ax = ax_list[i, j]
                ax.set_aspect(aspect, share=True)       # Set equal axes
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
    def images(images, **kwargs):
        """ Plot images from the first four observations, perfect, design and as many additional individual
        models as will fit in the grid.
        """
        # images, obs_dict = observations
        im_map = None
        png_path = kwargs.get('png_path', None)
        nrowcol = kwargs.get('nrowcol', (1, 1))
        shrink = kwargs.get('shrink', None)
        title = kwargs.get('title', '')
        pane_titles = kwargs.get('pane_titles', None)
        colourbar = kwargs.get('colourbar', True)

        n_rows, n_cols = nrowcol
        difference = kwargs.get('difference', False)
        ref_img = images[0]
        ny, nx = ref_img.shape
        # Only plot the central part of the image
        r1, r2, c1, c2 = 0, ny, 0, nx
        if shrink is not None:
            box_rad = int(0.5 * shrink * ny)
            r_cen, c_cen = int(ny / 2), int(nx / 2)
            r1, r2, c1, c2 = r_cen - box_rad, r_cen + box_rad + 1, c_cen - box_rad, c_cen + box_rad + 1

        if difference:
            img_diffs = []
            for img in images:
                img_diff = img - ref_img
                img_diffs.append(img_diff)
            images = img_diffs
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
            for img in images:
                vmin = vmin if vmin < np.amin(img) else np.amin(img)
                vmax = vmax if vmax > np.amax(img) else np.amax(img)

        for i, image in enumerate(images[0:4]):
            ax = ax_list[row, col]
            if pane_titles is not None:
                pane_title = pane_titles[pane]
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
            bar_label = 'Log$_{10}$(Signal)' if do_log else 'Signal'
            plt.colorbar(mappable=im_map, ax=ax_list, label=bar_label, shrink=0.75)
        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def _get_config_text(ds_dict):
        opticon = ds_dict['optical_configuration']
        field_no = ds_dict['field_no']
        slice_no = ds_dict['slice_no']
        spifu_no = ds_dict['spifu_no']
        wave = ds_dict['wavelength']
        text = ''
        common_text = ' $\lambda_{cov}$, $\lambda$ ='
        if opticon == 'nominal':
            fmt = "{:5.3f} $\mu$m, field={:d}, spat_slice={:d}"
            text = 'Nominal' + common_text + fmt.format(wave, field_no, slice_no)
        if opticon == 'spifu':
            fmt = "{:5.3f} $\mu$m, field={:d}, spat_slice={:d}, spec_slice={:d}"
            text = 'Extended' + common_text + fmt.format(wave, field_no, slice_no, spifu_no)
        return text

    @staticmethod
    def collage(image_list, obs_dict, **kwargs):
        png_path = kwargs.get('png_path', None)
        aspect = kwargs.get('aspect', 'auto')
        pane_titles = kwargs.get('pane_titles', ['Perfect', 'Design', 'M-C run 0', 'M-C run 1'])
        obs_title = ''
        if obs_dict is not None:
            obs_title = Plot._get_config_text(obs_dict)
        title = kwargs.get('title', obs_title)

        # observations = image_list, obs_dict
        Plot.images(image_list,
                    nrowcol=(2, 2), shrink=1.0, aspect=aspect,
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
        mc_percentiles = kwargs.get('mc_percentiles', None)

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
        ax_xlabel = data_par['xlabels'][axis]
        ax_xtag = 'w'

        y_perfect, y_design = ys[:, 0], ys[:, 1]
        y_mcs = ys[:, 2:]

        p_label, d_label, mc_label = 'perfect', 'design', '<M-C>'
        if type_key == 'lsf':
            fwhms = data['fwhm_gau']
            fmt = "{:s} {:4.2f}"
            p_label, d_label = fmt.format(p_label, fwhms[0]), fmt.format(d_label, fwhms[1])
            mc_mean_fwhm = np.mean(fwhms[2:])
            mc_label = fmt.format(mc_label, mc_mean_fwhm)

        y_plots = {'perfect': (p_label, 'red', 'solid', 2.0, 'o', y_perfect),
                   'design': (d_label, 'green', 'solid', 1.5, 'x', y_design)}
        y_plots, y_mc_lo, y_mc_hi = Plot._add_mc_percentile_profiles(y_mcs, mc_percentiles, y_plots, mc_label)
        ax.fill_between(x, y_mc_lo, y_mc_hi, color='skyblue')

        ax_ylabel = data_par['ylabels'][axis]
        if axis == 'radial':
            ax_ylabel = "Encircled energy fraction 'EE(r)'"
            ax_xlabel, ax_xtag = 'radius', 'r'
        ax.set_ylabel(ax_ylabel, fontsize=16.0)
        xunits = '(slices)' if ax_xlabel == 'across-slice' else '(det. pixels)'
        ax.set_xlabel("{:s} '{:s}' / {:s}".format(ax_xlabel, ax_xtag, xunits), fontsize=16.0)
        xlim = kwargs.get('xlim', None)
        if xlim is not None:
            ax.set_xlim(xlim)

        ax.set_title(title, fontsize=16.0)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        ax.xaxis.grid()
        ax.yaxis.grid()
        if xlog:
            ax.set_xscale('log')
        fillstyle = 'none'
        handles = []
        for key in y_plots:
            label, colour, ls, lw, marker, y = y_plots[key]
            handle, = ax.plot(x, y,
                              color=colour, marker=marker, mew=2.0, fillstyle=fillstyle,
                              ls=ls, lw=lw, label=label)
            if label is not None:
                handles.append(handle)
        ax.legend(handles=handles)

        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def _add_mc_percentile_profiles(y_mcs, mc_percentiles, y_plots, mc_mean_label):
        y_mc_lo, y_mc_hi = np.amin(y_mcs, axis=1), np.amax(y_mcs, axis=1)
        y_mc_mean = np.mean(y_mcs, axis=1)

        y_plots['<M-C>'] = mc_mean_label, 'blue', 'solid', 1.5, 'none', y_mc_mean
        y_plots['pc000'] = None, 'blue', 'solid', 0.5, 'none', y_mc_lo
        y_plots['pc100'] = None, 'blue', 'solid', 0.5, 'none', y_mc_hi

        if mc_percentiles is not None:
            y_mcs_sorted = np.sort(y_mcs, axis=1)
            n_pts, n_mcs = y_mcs_sorted.shape
            k = (n_mcs + 1) / 100.
            u_pcs = np.arange(0, n_mcs) * k

            for mc_pc, ls, lw in mc_percentiles:        # Add (non-zero and non-100) percentiles
                y_mc_pcs = np.zeros(n_pts)
                for i in range(0, n_pts):
                    y_pcs = y_mcs_sorted[i, :]
                    y_mc_pcs[i] = np.interp(mc_pc, u_pcs, y_pcs)
                label = "{:d} %ile".format(mc_pc)
                y_mc_pc_plot = label, 'blue', ls, lw, 'none', y_mc_pcs
                y_plots[label] = y_mc_pc_plot
        return y_plots, y_mc_lo, y_mc_hi

    @staticmethod
    def field_series(cube_series, is_spifu, **kwargs):
        png_path = kwargs.get('png_path', None)
        title = kwargs.get('title', '')
        select = kwargs.get('select', {})
        abscissa = kwargs.get('abscissa', ('wavelength', '$\mu$m'))
        ordinate = kwargs.get('ordinate', ('fwhms', 'pixels'))
        mc_percentiles = kwargs.get('mc_percentiles', None)
        nvals = len(cube_series['ipc_on'])
        sel_indices = np.full(nvals, True)

        for key in select:
            val = select[key]
            series = np.array(cube_series[key])
            sel_indices = np.logical_and(sel_indices, series == val)

        fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
        ax = ax_list
        ax.set_title(title)
        atext, aunit = abscissa
        xlabel = atext + ' ' + aunit
        ax.set_xlabel(xlabel, fontsize=16.0)
        otext, ounit = ordinate
        ylabel = otext + ' ' + ounit
        ax.set_ylabel(ylabel, fontsize=16.0)

        w_unsorted = np.array(cube_series['wavelength'])[sel_indices]
        sorted_indices = np.argsort(w_unsorted)
        plot_series = {}
        for key in cube_series:
            cube_array = np.array(cube_series[key])
            plot_array = cube_array[sel_indices]
            plot_series[key] = plot_array[sorted_indices]

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        ax.xaxis.grid()
        ax.yaxis.grid()

        x = plot_series['spifu_no'] if is_spifu else plot_series['wavelength']
        ys = plot_series['strehls'] if otext == 'strehls' else plot_series['fwhms']

        n_pts, n_profiles = ys.shape
        if otext == 'srps':
            dw_dlmspix = plot_series['dw_dlmspix']
            for i in range(0, n_pts):
                for j in range(0, n_profiles):
                    dw = ys[i, j] * dw_dlmspix[i]
                    ys[i, j] = x[i] * 1000. / dw

        y_perfect, y_design = ys[:, 0], ys[:, 1]
        y_mcs = ys[:, 2:]

        p_label, d_label, mc_label = 'perfect', 'design', '<M-C>'

        y_plots = {'perfect': (p_label, 'red', 'solid', 2.0, 'o', y_perfect),
                   'design': (d_label, 'green', 'solid', 1.5, 'x', y_design)}
        y_plots, y_mc_lo, y_mc_hi = Plot._add_mc_percentile_profiles(y_mcs, mc_percentiles, y_plots, mc_label)
        ax.fill_between(x, y_mc_lo, y_mc_hi, color='skyblue')

        handles = []
        for key in y_plots:
            y_plot = y_plots[key]
            label, colour, ls, lw, marker, y = y_plot
            handle, = ax.plot(x, y,
                              color=colour, marker=marker, mew=2.0,
                              ls=ls, lw=lw, label=label)
            if label is not None:
                handles.append(handle)
        ax.legend(handles=handles)

        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def phase_shift(data_type, data_tuple, ds_dict, ipc_tag, **kwargs):
        x, data = data_tuple

        png_path = kwargs.get('png_path', None)
        mc_percentiles = kwargs.get('mc_percentiles', None)
        normalise = kwargs.get('normalise', True)

        if normalise:
            data = data / np.mean(data, axis=0)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        xlabel = 'Image Shift [pix.]'
        ylabel = 'Value'
        ax.set_xlabel(xlabel, fontsize=16.0)
        ax.set_ylabel(ylabel, fontsize=16.0)

        title1 = Plot._get_config_text(ds_dict)
        fmt2 = "{:s} variation ({:s})"
        title2 = fmt2.format(data_type, ipc_tag)
        title = title1 + '\n' + title2
        ax.set_title(title)

        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)

        y_perfect, y_design = data[:, 0], data[:, 1]
        y_mcs = data[:, 2:]
        y_mc_mean = np.mean(y_mcs, axis=1)
        y_plots = [('perfect', 'red', 'solid', 2.0, 'o', y_perfect),
                   ('design', 'green', 'solid', 1.5, 'x', y_design),
                   ('<model>', 'blue', 'solid', 1.5, 'D', y_mc_mean)]
        if mc_percentiles is not None:
            y_plots, y_mc_lo, y_mc_hi = Plot._add_mc_percentile_profiles(y_mcs, mc_percentiles, y_plots)
            ax.fill_between(x, y_mc_lo, y_mc_hi, color='skyblue')

        handles = []
        for y_plot in y_plots:
            label, colour, ls, lw, marker, y = y_plot
            handle, = ax.plot(x, y,
                              color=colour, marker=marker, fillstyle='none', mew=2.0,
                              ls=ls, lw=lw, label=label)
            if label is not None:
                handles.append(handle)
        ax.legend(handles=handles)

        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return
