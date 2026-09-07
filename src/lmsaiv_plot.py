import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from lmsdist_util import Util
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
            if det_thetas[det_nos] is None:
                continue
            theta, theta_err = det_thetas[det_nos]
            title = r'$\theta_{:s}$ = {:8.3f}$\pm${:5.3f} deg.'.format(det_nos, theta, theta_err)

            ax.set_title(title)
            if pane == 2 or pane == 3:
                ax.set_xlabel('Row')
            if pane == 0 or pane == 2:
                ax.set_ylabel("Intra mosaic gap {:s}".format(det_nos))
            ax.plot(x, y, linestyle='none', marker='x', color='blue')
        plt.show()
        return

    @staticmethod
    def mosaic(mosaic, **kwargs):
        """ Plot the mosaic data structure (2 x 2 LMS images)
        :param mosaic:
        :param kwargs:  sb - Slice bounds (QTable format, det_no, slice_no, spifu_no, col, rowmin, rowmax)
                        cmap - Colour map for image
                        title - Plot title
                        overlay - Overlay graphical elements (iso-alpha/beta traces etc.)
        :return:
        """
        det_idx = None

        cmap_name = kwargs.get('cmap', 'hot')
        cmap = mpl.colormaps[cmap_name]
        sb = kwargs.get('sb', {})
        title = kwargs.get('title', '-')
        suptitle = mosaic.name + '\n' + title
        fig_size = 8, 7
        n_rows, n_cols = 2, 2
        ax_datas, ax_bounds = {}, {}
        overlay = kwargs.get('overlay', {})
        shareall = True                                     # Share axis scaling for all plots
        ax_bounds = {}                                      # Plot bounds for each axis
        axes_pad = 0.15, 0.15
        n_axes = 4
        if not overlay:                                     # An empty dictionary returns False
            for hdu in mosaic.hdu_list:
                det_no = int(hdu.header['ID'])
                ax_tag = "det_no={:d}".format(det_no)
                ax_bounds[ax_tag] = 0, 2048, 0, 2048
                ax_data = {'ax_idx': det_no - 1, 'image': hdu.data}
                ax_datas[ax_tag] = [ax_data]
        else:
            if overlay['type'] == 'det_traces':         # Make individual plots, 4 per row.
                # Each detector/pupil number is plotted in the same axis.
                is_iso_alpha = overlay['trace_type'] == 'iso_alpha'
                shareall = False
                axes_pad = 0.5, 0.5
                n_panes_row = 2
                det_traces = overlay['data']
                ax_idx_next = 0
                for det_trace in det_traces:
                    det_no = det_trace['det_no']
                    slice_no = det_trace['slice_no']
                    _, pslice_no = Util.decode_slice_no(slice_no)
                    ax_tag = "det_no={:d}, pslice={:d}".format(det_no, pslice_no)
                    det_idx = det_no - 1
                    v_fid = int(det_trace['v_fid'])
                    u_mean = int(det_trace['u_mean'])
                    ax_data = {'ax_idx': ax_idx_next, 'image': mosaic.hdu_list[det_idx].data, 'det_trace': det_trace}
                    if ax_tag in ax_datas:
                        ax_idx_copy = ax_datas[ax_tag][0]['ax_idx']
                        ax_data['ax_idx'] = ax_idx_copy
                        ax_datas[ax_tag].append(ax_data)
                        if is_iso_alpha:
                            x_min, x_max, y_min, y_max = 0, 2048, v_fid - 20, v_fid + 20
                        else:
                            old_bounds = ax_bounds[ax_tag]
                            x_min = min(old_bounds[0], v_fid - 10)
                            x_max = max(old_bounds[1], v_fid + 10)
                            y_min = min(old_bounds[2], u_mean - 80)
                            y_max = max(old_bounds[3], u_mean + 80)
                        ax_bounds[ax_tag] = x_min, x_max, y_min, y_max
                    else:
                        ax_datas[ax_tag] = [ax_data]
                        if is_iso_alpha:
                            ax_bounds[ax_tag] = 0, 2048, v_fid - 20, v_fid + 20
                        else:
                            ax_bounds[ax_tag] = v_fid - 10, v_fid + 10, u_mean - 80, u_mean + 80
                        ax_idx_next += 1
                n_axes = ax_idx_next
                n_rows, n_cols = (n_axes // (n_panes_row +1)) + 1, n_panes_row

        nrows_ncols = n_rows, n_cols
        if overlay is None:
            fig = plt.figure(figsize=fig_size)  # Set up figure and image grid
            grid = ImageGrid(fig, 111,
                             nrows_ncols=nrows_ncols, axes_pad=axes_pad, cbar_location="right",  share_all=shareall,
                             cbar_mode="single", cbar_size="7%", cbar_pad=0.15,
                             )
        else:
            n_rows, n_cols = nrows_ncols
            fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=fig_size,
                                     sharex='none', squeeze=True)
            n_panes = n_rows * n_cols
            for ax_idx in range(n_axes, n_panes):
                axes[ax_idx].remove()
            grid = axes.flatten()
        fig.suptitle(suptitle)

        # Find common signal bounds
        vmin, vmax = 1.E6, -1.E6
        for hdu in mosaic.hdu_list:
            vmin_hdu, vmax_hdu = np.nanmin(hdu.data), np.nanmax(hdu.data)
            vmin = min(vmin, vmin_hdu)
            vmax = max(vmax ,vmax_hdu)
        if 'vmin' in kwargs:
            vmin = kwargs.get('vmin', vmin)
        if 'vmax' in kwargs:
            vmax = kwargs.get('vmax', vmax)
        ax, im = None, None
        for ax_tag in ax_datas:
            ax_data_list = ax_datas[ax_tag]             # Get list of data sets to plot in this axis
            ax_idx = ax_data_list[0]['ax_idx']          # All data in list (should) share the same axis index.
            ax = grid[ax_idx]
            x_min, x_max, y_min, y_max = ax_bounds[ax_tag]
            ax.set_xlim(x_min - 1, x_max + 1)
            ax.set_ylim(y_min - 1, y_max + 1)
            aspect_ratio = (x_max - x_min) / (y_max - y_min)
            ax.set_aspect(aspect_ratio)

            for ax_data in ax_data_list:
                x1, x2, y1, y2 = int(x_min), int(x_max), int(y_min), int(y_max)
                image = ax_data['image']
                mask = kwargs.get('mask', None)
                if mask is not None:
                    mask_value, mask_colour = mask
                    image = np.ma.masked_where(image == mask_value, image)
                    cmap.set_bad(color=mask_colour)
                im = ax.imshow(image[y1:y2, x1:x2], extent=(x_min-.5, x_max+.5, y_min-1.5, y_max-.5),
                               interpolation='nearest', aspect=aspect_ratio,
                               cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                if sb:
                    det_no = sb['det_no']
                    idx = np.argwhere(det_no == det_idx + 1)
                    x = sb['det_col'][idx]
                    yrmin = sb['det_row_min'][idx]
                    ax.plot(x, yrmin, marker='o', ms=2.0, color='red', linestyle='none')
                    yrmax = sb['det_row_max'][idx]
                    ax.plot(x, yrmax, marker='o', ms=2.0, color='green', linestyle='none')
                # overlay = kwargs.get('overlay', None)
                if overlay:
                    if overlay['type'] == 'det_traces':
                        det_trace = ax_data['det_trace']
                        if det_trace is None:
                            continue
                        is_alpha = det_trace['type'] == 'iso-alpha'
                        pt_u_coords = det_trace['pt_u_coords']
                        pt_v_coords = det_trace['pt_v_coords']
                        xs = pt_u_coords if is_alpha else pt_v_coords
                        ys = pt_v_coords if is_alpha else pt_u_coords
                        # title = "{:d}, {:d}".format(det_trace['det_no'], det_trace['slice_no'])
                        ax.set_title(ax_tag)
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
        if overlay is None:
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

        for det_traces in [alpha_traces, lambda_traces]:
            for det_trace in det_traces:
                det_no = det_trace['det_no']
                ax = grid[det_no - 1]
                ax.set_xlim(xmin - 1, xmax + 1)
                ax.set_ylim(ymin - 1, ymax + 1)
                aspect_ratio = (xmax - xmin) / (ymax - ymin)
                ax.set_aspect(aspect_ratio)

                is_alpha = det_trace['type'] == 'iso-alpha'
                popt = det_trace['popt']
                xhw = 1024 if is_alpha else 40
                x1 = det_trace['u_mean'] - xhw
                x2 = x1 + 2 * xhw
                x = np.arange(x1, x2, 10)
                y = Globals.polynomial(x, *popt)
                colour = 'blue' if is_alpha else 'red'
                if is_alpha:
                    ax.plot(x, y, color=colour, linestyle='solid', lw=.5)
                else:
                    ax.plot(y, x, color=colour, linestyle='solid', lw=.5)
        coords = kwargs.get('coords', {})
        if coords:
            det_nos = np.array(coords['det_no'])
            rows = np.array(coords['row'])
            cols = np.array(coords['col'])
            for i in range(0, 4):
                idx = np.argwhere(det_nos == i+1)[:]
                x, y = cols[idx], rows[idx]
                ax = grid[i]
                ax.set_xlim(xmin - 1, xmax + 1)
                ax.set_ylim(ymin - 1, ymax + 1)
                aspect_ratio = (xmax - xmin) / (ymax - ymin)
                ax.set_aspect(aspect_ratio)
                ax.plot(x, y, marker='o', ms=2.0, color='black', linestyle='none')

        plt.show()
        return

    @staticmethod
    def histograms(mosaic):
        n_bins = 200
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, tight_layout=True)
        fig.suptitle(mosaic.name)

        for i, hdu in enumerate(mosaic.hdu_list):
            ax_row, ax_col = int(i / 2), i % 2
            vals = hdu.data.flatten()
            axs[ax_row, ax_col].hist(vals, bins=n_bins)
        plt.show()
        return

    @staticmethod
    def slice_gaps(gap_list):
        gaps = np.array(gap_list)
        figsize = [8, 8]
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize,
                                 sharex='all', sharey='all', squeeze=True)
        fig.suptitle('Intra-slice gaps')
        ax_list = np.array(axes).flatten()
        uni_det_nos = np.unique(gaps[:,0])
        for det_no in uni_det_nos:
            pane = int(det_no) - 1
            ax = ax_list[pane]
            ax.set_title('Det {:d}'.format(pane + 1))
            if pane == 2 or pane == 3:
                ax.set_xlabel('Slice no.')
            if pane == 0 or pane == 2:
                ax.set_ylabel('Gap / pixels')
            idx = np.argwhere(gaps[:, 0] == det_no)
            x, y = gaps[idx, 1], gaps[idx, 5]
            ax.plot(x, y, linestyle='none', marker='x', color='black')
        plt.show()
        return

    @staticmethod
    def profiles(profiles, nax_rows=2, nax_cols=2):
        """ Plot multiple profile tuples.
        """
        figsize = [8, 8]
        n_axes = nax_rows * nax_cols
        fig, axes = plt.subplots(nrows=nax_rows, ncols=nax_cols, figsize=figsize,
                                 sharex='all', sharey='all', squeeze=True)
        ax_list = [axes] if n_axes < 2 else axes
        ax_list = np.array(ax_list).flatten()
        for profile in profiles:
            label, colour, det_no, profile_column, vals, pts = profile
            iax = det_no - 1
            x = np.arange(len(vals))
            title = "Det {:d}".format(det_no)
            ax = ax_list[iax]
            ax.plot(x, vals, color=colour, linestyle='solid', lw=.5)
            ax.set_title(title)
            for pt in pts:
                x_pts, y_pts = pt
                ax.plot(x_pts, y_pts, linestyle='none', marker='x', color=colour)
        plt.show()
