import math
import numpy as np
import matplotlib.pyplot as plt


class Plot:

    def __init__(self):
        # plt.rcParams['backend'] = 'AGG'
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

        sharex = kwargs.get('sharex', xlim is not None)
        sharey = kwargs.get('sharey', ylim is not None)
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
        im_map = None
        png_path = kwargs.get('png_path', None)
        nrowcol = kwargs.get('nrowcol', (1, 1))
        shrink = kwargs.get('shrink', None)
        title = kwargs.get('title', '')
        pane_titles = kwargs.get('pane_titles', None)
        colourbar = kwargs.get('colourbar', True)
        vlim = kwargs.get('vlim', None)

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
        if vlim is None:
            vmin, vmax = np.finfo('float').max, np.finfo('float').min
            for img in images:
                vmin = vmin if vmin < np.amin(img) else np.amin(img)
                vmax = vmax if vmax > np.amax(img) else np.amax(img)
        else:
            vmin, vmax = vlim[0], vlim[1]

        for i, image in enumerate(images):
            ax = ax_list[row, col]
            if pane_titles is not None:
                pane_title = pane_titles[pane]
                ax.set_title(pane_title)

            if do_log:
                lvmax = math.log10(vmax)
                if vlim is None:
                    vmin = vmax / 1000.0
                lvmin = math.log10(vmin)
                clipped_image = np.where(image < vmin, vmin, image)
                log_image = np.log10(clipped_image)
                im_map = ax.imshow(log_image[r1:r2, c1:c2],
                                   extent=(c1-0.5, c2-0.5, r1-0.5, r2-0.5),
                                   vmin=lvmin, vmax=lvmax)
            else:
                if do_half and vlim is None:
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
        mc_bounds = ds_dict['mc_bounds']
        data_pars = {'ee': {'title_lead': 'Enslitted energy',
                            'xlabels': {'spectral': 'aperture half-width / det. pixels',
                                        'spatial': 'aperture half-height / det. pixels',
                                        'across-slice': 'aperture half-width / slices',
                                        'along-slice': 'aperture height / pixels'},
                            'ylabels': {'spectral': 'EE(x)',
                                        'spatial': 'EE(y)',
                                        'across-slice': 'EE(x)',
                                        'along-slice': 'EE(y)'},
                            },
                     'lsf': {'title_lead': 'Line profile',
                             'xlabels': {'spectral': 'spectral / det. pixels',
                                         'spatial': 'spatial / det. pixels',
                                         'across-slice': 'across slice / slices',
                                         'along-slice': 'along slice / det. pixels'},
                             'ylabels': {'spectral': 'response',
                                         'spatial': 'response',
                                         'across-slice': 'response',
                                         'along-slice': 'response'},
                             }
                     }
        data_par = data_pars[type_key]

        title_lead = data_par['title_lead']
        title = "{:s}\n{:s}, $\lambda$= {:5.3f} $\mu$m".format(title_lead, name, wavelength)

        fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
        ax = ax_list

        x = data['xvals']
        ys = data['yvals']
        xlabels = data_par['xlabels']
        ax_xlabel = xlabels[axis]

        y_perfect, y_design = ys[:, 0], ys[:, 1]
        y_mcs = ys[:, 2:]

        p_label, d_label, mc_label = 'perfect', 'design', '<M-C>'
        if type_key == 'lsf':
            fwhms = data['lin_fwhm']
            fmt = "{:s} {:4.2f}"
            p_label, d_label = fmt.format(p_label, fwhms[0]), fmt.format(d_label, fwhms[1])
            if mc_bounds is not None:
                mc_mean_fwhm = np.mean(fwhms[2:])
                mc_label = fmt.format(mc_label, mc_mean_fwhm)

        y_plots = {'perfect': (p_label, 'red', 'solid', 2.0, 'o', x, y_perfect),
                   'design': (d_label, 'green', 'solid', 1.5, 'x', x, y_design)}
        if mc_bounds is not None:
            y_plots, y_mc_lo, y_mc_hi = Plot._add_mc_percentile_profiles(x, y_mcs, mc_percentiles, y_plots, mc_label)
            ax.fill_between(x, y_mc_lo, y_mc_hi, color='skyblue')

        ax_ylabel = data_par['ylabels'][axis]
        if axis == 'radial':
            ax_ylabel = "Encircled energy fraction 'EE(r)'"
            ax_xlabel, ax_xtag = 'radius', 'r'
        ax.set_ylabel(ax_ylabel, fontsize=16.0)
        ax.set_xlabel("{:s}".format(ax_xlabel), fontsize=16.0)
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
            label, colour, ls, lw, marker, x, y = y_plots[key]
            handle, = ax.plot(x, y,
                              color=colour, marker=marker, mew=2.0, fillstyle=fillstyle,
                              ls=ls, lw=lw, label=label)
            if label is not None:
                handles.append(handle)
        ax.legend(handles=handles, prop={'size': 14.0})

        if png_path is not None:
            plt.savefig(png_path, bbox_inches='tight')
            plt.close(fig)
        plt.show()
        return

    @staticmethod
    def _add_mc_percentile_profiles(x, y_mcs, mc_percentiles, y_plots, mc_mean_label):
        y_mc_lo, y_mc_hi = np.amin(y_mcs, axis=1), np.amax(y_mcs, axis=1)
        y_mc_mean = np.mean(y_mcs, axis=1)

        y_plots['<M-C>'] = mc_mean_label, 'blue', 'solid', 1.5, 'none', x, y_mc_mean
        y_plots['pc000'] = None, 'blue', 'solid', 0.5, 'none', x, y_mc_lo
        y_plots['pc100'] = None, 'blue', 'solid', 0.5, 'none', x, y_mc_hi

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
                y_mc_pc_plot = label, 'blue', ls, lw, 'none', x, y_mc_pcs
                y_plots[label] = y_mc_pc_plot
        return y_plots, y_mc_lo, y_mc_hi

    @staticmethod
    def series(data, params):

        plot_key = params['plot_key']
        key_only = params['key_only']
        png_path = params['png_path']
        autox = params['xscale'] == 'auto'
        nrows, ncols, figsize = params['fig_layout']
        title = params['title']
        ordinate = params['ordinate']
        do_srp_rqt = params['do_srp_rqt']
        do_fwhm_rqt = params['do_fwhm_rqt']
        abscissa = params['abscissa']
        is_defocus = abscissa[0] == 'defocus'
        mc_percentiles = params['mc_percentiles']

        field_rowcols = {1: (1, 1), 2: (0, 1), 3: (2, 1),
                         4: (1, 0), 5: (0, 0), 6: (2, 0),
                         7: (1, 2), 8: (0, 2), 9: (2, 2),
                         10: (1, 1), 11: (0, 1), 12: (2, 1),
                         13: (1, 0), 14: (0, 0), 15: (2, 0),
                         16: (1, 2), 17: (0, 2), 18: (2, 2)}

        ykey, ylabel, ylim = ordinate
        xkey, xlabel, xlim = abscissa
        if autox:
            xlim = None

        fig, ax_list = Plot.set_plot_area(nrows=nrows, ncols=ncols, figsize=figsize,
                                          sharex=True, sharey=True,
                                          xlim=xlim, ylim=ylim)
        fig.suptitle(title)
        is_first_plot = True
        for data_key in data:
            field_data = data[data_key]
            field_no = field_data['field_no']
            field_label = "Field {:d}".format(field_no)
            is_spifu = field_data['is_spifu']
            mc_bounds = field_data['mc_bounds']
            row, col = field_rowcols[field_no]
            col = col - 1 if is_spifu else col
            ax = ax_list[row, col]
            if row == nrows - 1:
                ax.set_xlabel(xlabel)
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.set_title(field_label)
            if not key_only:
                ax.xaxis.grid()
                ax.yaxis.grid()

            x_unsort = np.array(field_data['x_values'])
            indices = np.argsort(x_unsort)
            ys_unsort = np.array(field_data[ykey])
            fs_unsort = np.array(field_data['focus_shifts'])

            x = x_unsort[indices]
            ys = ys_unsort[indices, :]
            fs = fs_unsort[indices]
            y_perfect, y_design = ys[:, 0], ys[:, 1]
            if mc_bounds is not None:
                y_mcs = ys[:, 2:]

            p_label, d_label = 'perfect', 'design'
            y_plots = {'perfect': (p_label, 'red', 'solid', 2.0, 'o', x, y_perfect)}
            if is_defocus:
                defoci = set(fs)
                symbols = {0: 'o', 50:'1', 100:'x', 200:'+'}
                for i, defoc in enumerate(defoci):
                    indices = np.argwhere(fs == defoc)
                    y_df = y_design[indices]
                    x_df = x[indices]
                    key = "des_defoc_{:d}um".format(defoc)
                    sym = symbols[defoc]
                    y_plots[key] = (key, 'green', 'dashed', 1.5, sym, x_df, y_df)
            else:
                y_plots['design'] = (d_label, 'green', 'solid', 1.5, '.', x, y_design)
            if mc_bounds is not None:
                mc_label = '<M-C>'
                y_plots, y_mc_lo, y_mc_hi = Plot._add_mc_percentile_profiles(x, y_mcs, mc_percentiles, y_plots, mc_label)
                if not key_only:
                    ax.fill_between(x, y_mc_lo, y_mc_hi, color='skyblue')

            handles = []
            for pkey in y_plots:
                y_plot = y_plots[pkey]
                label, colour, ls, lw, marker, x, y = y_plot
                if key_only:
                    x, y = [0.], [0.]
                handle, = ax.plot(x, y,
                                  color=colour, marker=marker, mew=2.0,
                                  ls=ls, lw=lw, label=label)
                if label is not None:
                    handles.append(handle)
            if do_srp_rqt:
                x_srp = np.array(x)
                y_srp = np.full(y.shape, 1E5)
                idx = np.argwhere(x_srp > 4.8)
                y_srp[idx] = 0.85E5
                label = 'Rqt. MET-2745 SRP'
                handle, = ax.plot(x_srp, y_srp,
                                  color='purple', marker='none', mew=2.0,
                                  ls='dotted', lw=1.5, label=label)
                handles.append(handle)
            if do_fwhm_rqt:
                x_rqt = np.array(x)
                y_rqt = np.full(len(x), 2.0)
                label = 'Rqt. MET-3739 FWHM'
                handle, = ax.plot(x_rqt, y_rqt,
                                  color='purple', marker='none', mew=2.0,
                                  ls='dotted', lw=1.5, label=label)
                handles.append(handle)
            if is_first_plot and (plot_key or key_only):
                ax.legend(handles=handles)
                is_first_plot = False

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
