import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes
from mpl_toolkits.axes_grid1 import ImageGrid


class Plot:

    def __init__(self):
        return

    @staticmethod
    def mosaic(mosaic, **kwargs):
        file_name, hdr, data = mosaic

        # Set up figure and image grid
        fig = plt.figure(figsize=(8, 7))
        fig.suptitle(file_name)
        grid = ImageGrid(fig, 111,
                         nrows_ncols=(2, 2),
                         axes_pad=0.15,
                         share_all=True,
                         cbar_location="right",
                         cbar_mode="single",
                         cbar_size="7%",
                         cbar_pad=0.15,
                         )

        xmin, xmax = 0, data[0].shape[0]
        ymin, ymax = 0, data[0].shape[1]
        vmin = kwargs.get('vmin', np.nanmin(data))
        vmax = kwargs.get('vmax', np.nanmax(data))
        for det_idx, ax in enumerate(grid):
            ax.set_xlim(xmin-1, xmax+1)
            ax.set_ylim(ymin-1, ymax+1)
            ax.set_aspect('equal', 'box')
            image = data[det_idx]
            im = ax.imshow(image, extent=(xmin-0.5, xmax+0.5, ymin-0.5, ymax+0.5),
                           interpolation='nearest', cmap='hot', vmin=vmin, vmax=vmax, origin='lower')
        ax.cax.colorbar(im)
        plt.show()
        return

    @staticmethod
    def histograms(mosaics):
        for mosaic in mosaics:
            file_name, hdr, data = mosaic
            n_bins = 200
            fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, tight_layout=True)
            fig.suptitle(file_name)

            for i, image in enumerate(data):
                ax_row, ax_col = int(i / 2), i % 2
                vals = image.flatten()
                axs[ax_row, ax_col].hist(vals, bins=n_bins)
            plt.show()
        return

    @staticmethod
    def profiles(profiles, nax_rows=1, nax_cols=1):
        """ Plot multiple profile tuples.
        """
        figsize = [8, 8]
        n_axes = nax_rows * nax_cols
        fig, axes = plt.subplots(nrows=nax_rows, ncols=nax_cols, figsize=figsize,
                                 sharex='all', sharey='all', squeeze=True)
        ax_list = axes if n_axes > 1 else [axes]
        ax_list = np.array(ax_list).flatten()
        for i, profile in enumerate(profiles):
            title, x_val, y_val, pts_list = profile

            ax = ax_list[i]
            ax.plot(x_val, y_val)
            ax.set_title(title)
            for pts in pts_list:
                x_pts, y_pts, colour = pts
                # n_on = len(on_rows)
                # mark_y = [cut_level] * n_on
                ax.plot(x_pts, y_pts, linestyle='none', marker='x', color=colour)
                # ax.plot(off_rows, mark_y, linestyle='none', marker='+', color='red')

        plt.show()
