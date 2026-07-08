import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from lms_globals import Globals


class Plot:

    def __init__(self):
        return

    @staticmethod
    def gap_data(gap_data, det_thetas):
        figsize = [8, 8]
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize,
                                 sharex='all', squeeze=True)

        for pane, det_nos in enumerate(['12', '34']):
            ax = axes[pane]
            x, y = [], []
            for slice_no in gap_data:
                gap = gap_data[slice_no]
                if str(gap['det_no']) in det_nos:
                    x.append(gap['u_mean'])
                    y.append(gap['col_gap'])
            # tag = r'{:s}'.format(det_nos)
            theta, theta_err = det_thetas[det_nos]
            title = r'$\theta_{:s}$ = {:8.3f}$\pm${:5.3f} deg.'.format(det_nos, theta, theta_err)

            ax.set_title(title)
            ax.set_xlabel('Row')
            ax.set_ylabel("Intra mosaic gap {:s}".format(det_nos))
            ax.plot(x, y, linestyle='none', marker='x', color='blue')
        plt.show()
        return

    @staticmethod
    def mosaic(mosaic, **kwargs):
        """ Plot the mosaic data structure (2 x 2 LMS images)
        :param mosaic:
        :param kwargs:
        :return:
        """
        file_name, primary_hdr, hdus = mosaic
        cmap_name = kwargs.get('cmap', 'hot')
        cmap = mpl.colormaps[cmap_name]
        sb = kwargs.get('sb', None)         # Slice bounds (QTable format, det_no, slice_no, spifu_no, col, rowmin, rowmax)
        title = kwargs.get('title', '-')
        suptitle = file_name + '\n' + title
        # Set up figure and image grid
        fig = plt.figure(figsize=(8, 7))
        fig.suptitle(suptitle)
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(2, 2), axes_pad=(0.15, 0.15), cbar_location="right",  share_all=True,
                         cbar_mode="single", cbar_size="7%", cbar_pad=0.15,
                         )
        # Set plot limits
        xmin, xmax = 0, hdus[0].shape[1]
        ymin, ymax = 0, hdus[0].shape[0]
        bounds = kwargs.get('bounds', (xmin, xmax, ymin, ymax))
        xmin, xmax, ymin, ymax = bounds

        vmin, vmax = 1.E6, -1.E6
        for hdu in hdus:
            vmin_hdu, vmax_hdu = np.nanmin(hdu.data), np.nanmax(hdu.data)
            vmin = min(vmin, vmin_hdu)
            vmax = max(vmax ,vmax_hdu)
        if 'vmin' in kwargs:
            vmin = kwargs.get('vmin', vmin)
        if 'vmax' in kwargs:
            vmax = kwargs.get('vmax', vmax)
        ax, im = None, None
        data_origin = primary_hdr['ORIGIN']
        is_toysim = 'TOYSIM' in data_origin
        for hdu in hdus:
            det_no = int(hdu.header['ID'])
            det_idx = det_no - 1 if is_toysim else Globals.mos_idx[det_no]
            ax = grid[det_idx]
            ax.set_xlim(xmin-1, xmax+1)
            ax.set_ylim(ymin-1, ymax+1)
            aspect_ratio = (xmax-xmin)/(ymax-ymin)
            ax.set_aspect(aspect_ratio)
            x1, x2, y1, y2 = int(xmin), int(xmax), int(ymin), int(ymax)
            image = hdus[det_idx].data
            mask = kwargs.get('mask', None)
            if mask is not None:
                mask_value, mask_colour = mask
                image = np.ma.masked_where(image == mask_value, image)
                cmap.set_bad(color=mask_colour)
            im = ax.imshow(image[y1:y2, x1:x2], extent=(xmin-.5, xmax+.5, ymin-1.5, ymax-.5),
                           interpolation='nearest', aspect=aspect_ratio,
                           cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
            if sb is not None:
                det_no = sb['det_no']
                idx = np.argwhere(det_no == det_idx + 1)
                x = sb['det_col'][idx]
                yrmin = sb['det_row_min'][idx]
                ax.plot(x, yrmin, marker='o', ms=2.0, color='red', linestyle='none')
                yrmax = sb['det_row_max'][idx]
                ax.plot(x, yrmax, marker='o', ms=2.0, color='green', linestyle='none')
            overlay = kwargs.get('overlay', None)
            if overlay is not None:
                if overlay['type'] == 'det_traces':
                    trace_data = overlay['data']
                    for det_trace in trace_data:
                        if det_trace['det_no'] == det_no:
                            is_alpha = det_trace['type'] == 'iso-alpha'
                            pt_u_coords = det_trace['pt_u_coords']
                            pt_v_coords = det_trace['pt_v_coords']
                            xs = pt_u_coords if is_alpha else pt_v_coords
                            ys = pt_v_coords if is_alpha else pt_u_coords
                            ax.plot(xs, ys, marker='o', ms=4.0, color='cyan', linestyle='none')
                            popt = det_trace['popt']
                            xhw = 1024 if is_alpha else 40
                            x1 = det_trace['u_mean'] - xhw
                            x2 = x1 + 2 * xhw
                            x = np.arange(x1, x2, 10)
                            y = Globals.polynomial(x, *popt)
                            if is_alpha:
                                ax.plot(x, y, color='blue', linestyle='solid')
                            else:
                                ax.plot(y, x, color='blue', linestyle='solid')
        ax.cax.colorbar(im)
        plt.show()
        return

    @staticmethod
    def det_traces(alpha_traces, lambda_traces, config_id, **kwargs):
        fig = plt.figure(figsize=(8, 7))

        fig.suptitle('Configuration ' + config_id)
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(2, 2), axes_pad=(0.15, 0.15), share_all=True
                         )
        # Set plot limits
        xmin, xmax = 0, Globals.det_format[1]
        ymin, ymax = 0, Globals.det_format[0]
        bounds = kwargs.get('bounds', (xmin, xmax, ymin, ymax))
        xmin, xmax, ymin, ymax = bounds

        alpha_stretch = kwargs.get('alpha_stretch', 1.)
        ybar = {}

        for det_traces in [alpha_traces, lambda_traces]:
            for det_trace in det_traces:
                det_no = det_trace['det_no']
                ax = grid[det_no - 1]

                is_alpha = det_trace['type'] == 'iso-alpha'
                popt = det_trace['popt']
                xhw = 1024 if is_alpha else 40
                x1 = det_trace['u_mean'] - xhw
                x2 = x1 + 2 * xhw
                x = np.arange(x1, x2, 10)
                y = Globals.polynomial(x, *popt)
                if is_alpha:
                    slice_no = det_trace['slice_no']
                    if slice_no not in ybar:        # Use the first y_bar value found for this slice.
                        ybar[slice_no] = np.mean(y)
                    y = y + (y - ybar[slice_no]) * alpha_stretch
                    ax.plot(x, y, color='blue', linestyle='solid', lw=.5)
                else:
                    ax.plot(y, x, color='orange', linestyle='solid', lw=.5)
        coords = kwargs.get('coords', None)
        if coords is not None:
            det_nos = coords['det_no']
            rows = coords['row']
            cols = coords['col']
            for i in range(0, len(det_nos)):
                det_no = det_nos[i]
                ax = grid[det_no - 1]
                x, y = cols[i], rows[i]
                ax.plot(x, y, marker='o', ms=2.0, color='black', linestyle='none')

        plt.show()
        return

    @staticmethod
    def histograms(mosaic):
        file_name, hdr, hdus = mosaic
        n_bins = 200
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, tight_layout=True)
        fig.suptitle(file_name)

        for i, hdu in enumerate(hdus):
            ax_row, ax_col = int(i / 2), i % 2
            vals = hdu.data.flatten()
            axs[ax_row, ax_col].hist(vals, bins=n_bins)
        plt.show()
        return

    @staticmethod
    def profiles(profiles, nax_rows=2, nax_cols=2):
        """ Plot multiple profile tuples.
        """
        figsize = [8, 8]
        n_profiles = len(profiles)
        # nax_rows = Globals.n_lms_detectors
        # nax_cols = int(n_profiles / nax_rows)

        n_axes = nax_rows * nax_cols
        fig, axes = plt.subplots(nrows=nax_rows, ncols=nax_cols, figsize=figsize,
                                 sharex='all', sharey='all', squeeze=True)
        ax_list = [axes] if n_axes < 2 else axes
        ax_list = np.array(ax_list).flatten()
        for profile in profiles:
            label, det_no, profile_column, vals, pts = profile
            iax = det_no - 1
            x = np.arange(len(vals))
            title = "Det {:d}".format(det_no)
            ax = ax_list[iax]
            ax.plot(x, vals)
            ax.set_title(title)
            for pt in pts:
                x_pts, y_pts = pt
                ax.plot(x_pts, y_pts, linestyle='none', marker='x')
        plt.show()
