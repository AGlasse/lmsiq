import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes
from mpl_toolkits.axes_grid1 import ImageGrid

from lms_globals import Globals


class Plot:

    def __init__(self):
        return

    @staticmethod
    def mosaic(mosaic, **kwargs):
        file_name, hdr, hdus = mosaic

        cmap_name = kwargs.get('cmap', 'hot')
        cmap = mpl.colormaps[cmap_name]
        sb = kwargs.get('sb', None)         # Slice bounds (QTable format, det_no, slice_no, spifu_no, col, rowmin, rowmax)
        title = kwargs.get('title', file_name)

        # Set up figure and image grid
        fig = plt.figure(figsize=(8, 7))
        fig.suptitle(title)
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(2, 2),
                         axes_pad=(0.15, 0.15),
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )
        # Set plot limits
        xmin, xmax = 0, hdus[0].shape[0]
        ymin, ymax = 0, hdus[0].shape[1]
        bounds = kwargs.get('bounds', (xmin, xmax, ymin, ymax))
        xmin, xmax, ymin, ymax = bounds

        vmin, vmax = 1.E6, -1.E6
        for hdu in hdus:
            vmin_hdu, vmax_hdu = np.nanmin(hdu.data), np.nanmax(hdu.data)
            vmin = vmin if vmin < vmin_hdu else vmin_hdu
            vmax = vmax if vmax > vmax_hdu else vmax_hdu
        if 'vmin' in kwargs:
            vmin = kwargs.get('vmin', np.nanmin(hdus))
        if 'vmax' in kwargs:
            vmax = kwargs.get('vmax', np.nanmax(hdus))
        ax, im = None, None
        for hdu in hdus:
            det_no = int(hdu.header['ID'])
            det_idx = Globals.mos_idx[det_no]
            # det_idx = det_no - 1
            ax = grid[det_idx]
            ax.set_xlim(xmin-1, xmax+1)
            ax.set_ylim(ymin-1, ymax+1)
            # ax.set_title("SS_DET_{:d}".format(det_no))
            ax.set_aspect('equal', 'box')
            image = hdus[det_idx].data
            mask = kwargs.get('mask', None)
            if mask is not None:
                mask_value, mask_colour = mask
                image = np.ma.masked_where(image == mask_value, image)
                cmap.set_bad(color=mask_colour)

            im = ax.imshow(image, extent=(xmin-0.5, xmax+0.5, ymin-0.5, ymax+0.5),
                           interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
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
                is_alpha = overlay['type'] == 'alpha'
                det_nos = np.array(overlay['det_no'])
                indices = np.argwhere(det_no == det_nos)
                if len(indices) < 1:
                    continue

                for idx in indices[:, 0]:
                    pt_u_coords = overlay['pt_u_coords'][idx]
                    pt_v_coords = overlay['pt_v_coords'][idx]
                    xs = pt_u_coords if is_alpha else pt_v_coords
                    ys = pt_v_coords if is_alpha else pt_u_coords
                    # ys_fit = Globals.cubic(xs, *popts[idx]) if is_alpha else Globals.cubic(ys, *popts)
                    ax.plot(xs, ys, marker='o', ms=2.0, color='red', linestyle='none')
        ax.cax.colorbar(im)

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
    def profiles(profiles, nax_rows=1, nax_cols=1):
        """ Plot multiple profile tuples.
        """
        figsize = [8, 8]
        n_profiles = len(profiles)
        nax_rows = Globals.n_lms_detectors
        nax_cols = int(n_profiles / nax_rows)

        n_axes = nax_rows * nax_cols
        fig, axes = plt.subplots(nrows=nax_rows, ncols=nax_cols, figsize=figsize,
                                 sharex='all', sharey='all', squeeze=True)
        ax_list = [axes] if n_axes < 2 else axes
        ax_list = np.array(ax_list).flatten()
        for i, profile in enumerate(profiles):
            title, x_val, y_val, pts_list = profile
            ax = ax_list[i]
            ax.plot(x_val, y_val)
            ax.set_title(title)
            for pts in pts_list:
                x_pts, y_pts, colour = pts
                ax.plot(x_pts, y_pts, linestyle='none', marker='x', color=colour)
        plt.show()
