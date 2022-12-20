import math
import numpy as np
import matplotlib.pyplot as plt


class LMSIQPlot:

    def __init__(self):
        return

    @staticmethod
    def images(observations, **kwargs):
        """ Plot images from the first four observations (perfect, design and as many additional individual
        models as will fit in the grid.
        """
        nrowcol = kwargs.get('nrowcol', (3, 5))
        title = kwargs.get('title', '')
        pane_titles = kwargs.get('pane_titles', None)
        plotregion = kwargs.get('plotregion', 'all')
        n_rows, n_cols = nrowcol
        fig, ax_list = plt.subplots(n_rows, n_cols, figsize=(10, 8))
        ax_list = np.atleast_2d(ax_list)

        fig.suptitle(title)
        pane, row, col = 0, 0, 0
        vmin, vmax, lvmin, lvmax = None, None, None, None
        do_log = True
        do_half = False
        box_rad = 20 if do_log else 8
        first_image = True
        for observation in observations:
            image, params = observation
            file_id, _ = params
            ax = ax_list[row, col]
            if pane_titles is not None:
                pane_title = file_id if pane_titles == 'file_id' else pane_titles[pane]
                ax.set_title(pane_title)
            if first_image:
                ny, nx = image.shape
                r1, r2, c1, c2 = 0, ny-1, 0, nx-1
                # Only plot the central part of the image
                if plotregion == 'centre':
                    r_cen, c_cen = int(ny/2), int(nx/2)
                    r1, r2, c1, c2 = r_cen - box_rad, r_cen + box_rad, c_cen - box_rad, c_cen + box_rad
                vmin, vmax = np.amin(image), np.amax(image)
                if do_log:
                    lvmax = math.log10(vmax)
                    vmin = vmax / 1000.0
                    lvmin = math.log10(vmin)
                first_image = False
            if do_log:
                clipped_image = np.where(image < vmin, vmin, image)
                log_image = np.log10(clipped_image)
                ax.imshow(log_image[r1:r2, c1:c2], vmin=lvmin, vmax=lvmax)
            if do_half:
                vmax = np.amax(image)
                vmin = vmax / 2.0
                ax.imshow(image[r1:r2, c1:c2], vmin=vmin, vmax=vmax)
            pane += 1
            if n_rows > 1 and n_cols > 1:
                col += 1
                if col == n_cols:
                    col = 0
                    row += 1
            else:
                col = pane
        plt.show()
        return

    @staticmethod
    def plot_ee(ee_data, wav, x_ref, ee_refs, ipc_factor, **kwargs):
        plot_all = kwargs.get('plot_all', True)

        key_list = []
        folder, axis, xlms, y_mean, y_rms, y_all = ee_data
        fmt = "{:7.3f} um, IPC = {:6.3f}"
        title = folder + '-' + axis + r'  $\lambda$ =' + fmt.format(wav, ipc_factor)

        fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
        ax = ax_list
        n_points, n_files = y_all.shape

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
        ax.set_ylabel("En-slitted energy fraction 'EE({:s})'".format(ax_xtag), fontsize=16.0)
        ax.set_xlabel("Aperture {:s} '{:s}' (pixels)".format(ax_xlabel, ax_xtag), fontsize=16.0)
        ax.set_title(title, fontsize=16.0)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        ax.xaxis.grid()
        ax.yaxis.grid()

        if plot_all:
            for j in range(0, n_files):
                y = y_all[:, j]
                ax.plot(a, y, color='grey', lw=0.5)

        col = 'red'
        ee_ref = ee_refs[1]
        ax.plot(a, y_all[:, 0], color=col)
        ax.plot(a_ref, ee_ref, color=col, marker='o')
        key_list.append(('perfect', col, ee_ref, None))

        col = 'green'
        ee_ref = ee_refs[2]
        ax.plot(a, y_all[:, 1], color=col)
        ax.plot(a_ref, ee_ref, color=col, marker='o')
        key_list.append(('design', col, ee_ref, None))

        col = 'blue'
        ee_ref = ee_refs[0]
        ax.plot(a, y_mean, color=col, lw=2.0)
        ax.plot(a_ref, ee_refs[0], color=col, marker='o')
        key_list.append(('<model>', col, ee_ref, None))
        ee_title = "EE at '{:s}'= {:5.2f} pix.".format(ax_xtag, 2 * x_ref)

        LMSIQPlot.key(ax, key_list, ee_title, xlim, ylim, fmts=['{:>8.3f}', None], col_labels=None)

        plt.show()
        return

    @staticmethod
    def find_hwhm(x, y):
        ymax = np.amax(y)
        yh = ymax / 2.0
        yz = np.subtract(y, yh)       # Search for zero crossing
        iz = np.where(yz > 0)
        il = iz[0][0]
        ir = iz[0][-1]
        xl = x[il-1] - yz[il-1] * (x[il] - x[il-1]) / (yz[il] - yz[il-1])
        xr = x[ir] - yz[ir] * (x[ir+1] - x[ir]) / (yz[ir+1] - yz[ir])
        return xl, xr, yh

    @staticmethod
    def plot_lsf(lsf_data, wav, dw_pix, ipc_factor, **kwargs):
        """ Plot line spread functions.  y_all element of lsf_data holds the perfect and esign profiles
        in index 0 and 1,
        :param lsf_data:
        :param wav:
        :param dw_pix:
        :param kwargs:
        :return:
        """
        plot_all = kwargs.get('plot_all', True)
        hwlim = kwargs.get('hwlim', 6.0)        # HWHM of 'perfect' profile plot limit

        folder, axis, x, y_mean, y_rms, y_all = lsf_data
        title = folder + '-' + axis + r'  $\lambda$ =' + "{:7.3f} $\mu$m, IPC = {:6.3f}".format(wav, ipc_factor)

        ynorm = np.amax(y_all[:, 0])        # Normalise to peak of perfect LSF
        y_perf = np.divide(y_all[:, 0], ynorm)
        xl, xr, yh = LMSIQPlot.find_hwhm(x, y_perf)
        xhw = 0.5 * (xr - xl)
        hwpix = int(hwlim*xhw + 1)
        xlim = [-hwpix, hwpix]
        ylim = [0.0, 1.05]

        xyarrow = 0.1
        xarrowlen = xyarrow * 6.0 * xhw
        yarrowlen = xyarrow * yh

        fig, ax_list = plt.subplots(1, 1, figsize=(10, 8))
        ax = ax_list
        n_points, n_files = y_all.shape

        xtick_interval = 1.0 if xlim[1] < 5.0 else 2.0
        xtick_lin_vals = np.arange(xlim[0], xlim[1]+1.0, xtick_interval)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xticks(xtick_lin_vals)
        ax.set_xticklabels(xtick_lin_vals)
        ax_ylabel = 'Spectral'
        ax_xtag = 'x'
        if axis == 'spatial':
            ax_ylabel = 'Spatial'
            ax_xtag = 'y'

        font_size = 16.0
        ax.set_ylabel("{:s} Profile 'f({:s})'".format(ax_ylabel, ax_xtag), fontsize=font_size)
        ax.set_xlabel("'{:s}' (pixels)".format(ax_xtag), fontsize=font_size)
        ax.set_title(title, fontsize=font_size)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(font_size)
        if plot_all:
            for j in range(2, n_files):
                y = y_all[:, j]
                ynorm = np.amax(y)
                y = np.divide(y, ynorm)
                ax.plot(x, y, color='grey', lw=0.5)

        key_list = []
        col = 'red'
        ynorm = np.amax(y_all[:, 0])
        y_perf = np.divide(y_all[:, 0], ynorm)
        ax.plot(x, y_perf, color=col, lw=2.0)
        LMSIQPlot.hwarrow(ax, xl, xr, yh, xarrowlen, yarrowlen, col)
        srp = wav / (dw_pix * (xr - xl))
        xfwhm = xr - xl
        key_list.append(('perfect', col, srp, xfwhm))

        col = 'green'
        ynorm = np.amax(y_all[:, 1])
        y_des = np.divide(y_all[:, 1], ynorm)
        xl, xr, yh = LMSIQPlot.find_hwhm(x, y_des)
        ax.plot(x, y_des, color=col, lw=2.0, marker='o')
        LMSIQPlot.hwarrow(ax, xl, xr, yh, xarrowlen, yarrowlen, col)
        srp = wav / (dw_pix * (xr - xl))
        xfwhm = xr - xl
        key_list.append(('design', col, srp, xfwhm))

        col = 'blue'
        ynorm = np.amax(y_mean)
        y_mod = np.divide(y_mean, ynorm)
        xl, xr, yh = LMSIQPlot.find_hwhm(x, y_mod)
        ax.plot(x, y_mod, color=col, lw=2.0, marker='o')
        LMSIQPlot.hwarrow(ax, xl, xr, yh, xarrowlen, yarrowlen, col)
        srp = wav / (dw_pix * (xr - xl))
        xfwhm = xr - xl
        key_list.append(('<model>', col, srp, xfwhm))

        if axis == 'spectral':
            LMSIQPlot.key(ax, key_list, "Spectral Resolving Power", xlim, ylim, fmts=["{:>8.0f}", "{:>8.2f}"])
        plt.show()
        return

    @staticmethod
    def profiles(profiles_list, **kwargs):
        config = kwargs.get('config', None)
        config_idxs = {'srp': 3, 'strehl': 4, 'fwhmspec': 5, 'fwhmspat': 6}
        idx = config_idxs[config]
        colours = ['green', 'red', 'blue', 'black']
        xlabel = "Wavelength [$\mu$m]"
        ylabel = kwargs.get('ylabel', 'Value')
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.set_xlabel(xlabel, fontsize=16.0)
        ax.set_ylabel(ylabel, fontsize=16.0)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(16.0)

        marker = '+'
        for i, profiles in enumerate(profiles_list):
            colour = colours[i]
            w = profiles[1]
            ipc = profiles[2][1]
            label = "IPC = {:10.3f}".format(ipc)
            y, yerr = profiles[idx]
            ax.plot(w, y, lw=1.0, marker=marker, mew=2.0, label=label, color=colour)
            if ipc == 0.0:
                ylo, yhi = np.array(y) - np.array(yerr), np.array(y) + np.array(yerr)
                ax.plot(w, ylo, lw=1.0, ls='dotted', color=colour)
                ax.plot(w, yhi, lw=1.0, ls='dotted', color=colour)
        plt.legend()
        plt.show()
        return

    @staticmethod
    def val_v_wav(w, y_list, **kwargs):
        xlabel = 'Wavelength [$\mu$m]'
        ylabel = kwargs.get('ylabel', 'Value')
        n_plots = len(y_list)
        errs = kwargs.get('errs', None)
        colours = kwargs.get('colours', ['blue']*n_plots)
        labels = kwargs.get('labels', ['label']*n_plots)
        markers = kwargs.get('markers', ['o']*n_plots)
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for i in range(0, n_plots):
            y, color, label, marker = y_list[i], colours[i], labels[i], markers[i]
            if errs is None:
                ax.plot(w, y, color=color, lw=1.0, marker=marker, mew=2.0, label=label)
            else:
                err = errs[i]
                ax.errorbar(w, y, yerr=err, lw=1.0, marker=marker, mew=2.0, label=label)
        plt.legend()
        plt.show()
        return

    @staticmethod
    def key(ax, key_list, title, xlim, ylim, **kwargs):
        fmts = kwargs.get('fmts', ['{:>6.2f}', '{:>6.2f}'])
        col_labels = kwargs.get('col_labels', ['SRP', 'FWHM/pix.'])

        xr, yr = xlim[1] - xlim[0], ylim[1] - ylim[0]
        x1 = xlim[0] + 0.02 * xr
        x2 = x1 + 0.15 * xr
        x3 = x2 + 0.15 * xr
        y = ylim[1] - 0.04 * yr
        ax.text(x1, y, title, color='black', fontsize=16.0)
        if col_labels is not None:
            y -= 0.05
            ax.text(x2, y, col_labels[0], color='black', fontsize=16.0)
            ax.text(x3, y, col_labels[1], color='black', fontsize=16.0)
        for key in key_list:
            y -= 0.05 * yr
            text, colour, srp, fwhm = key
            text_params = {'color': colour, 'fontsize': 16.0}
            lab = "{:<8s}".format(text)
            ax.text(x1, y, lab, **text_params)
            num = fmts[0].format(srp)
            ax.text(x2, y, num, **text_params)
            if fwhm is not None:
                num = fmts[1].format(fwhm)
                ax.text(x3, y, num, **text_params)
        return

    @staticmethod
    def hwarrow(ax, xl, xr, yh, xlen, ylen, colour):
        xs = [xr, xr+xlen, xr+0.3*xlen, xr, xr+0.3*xlen, xl-0.3*xlen, xl, xl-0.3*xlen]
        ys = [yh,      yh, yh+0.3*ylen, yh, yh-0.3*ylen, yh+0.3*ylen, yh, yh-0.3*ylen]

        kwargs = {'color': colour, 'lw': 1.0}
        ax.plot(xs[0:2], ys[0:2], **kwargs)
        ax.plot(xs[2:5], ys[2:5], **kwargs)
        ax.plot(xs[5:8], ys[5:8], **kwargs)
        return
