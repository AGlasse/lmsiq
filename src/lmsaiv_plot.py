import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


class Plot:

    def __init__(self):
        return

    @staticmethod
    def mosaic(mosaic, **kwargs):
        file_name, hdr, data = mosaic

        # Set up figure and image grid
        fig = plt.figure(figsize=(8, 7))
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
    def profiles(profiles):
        n_profiles = len(profiles)
        figsize = [8, 8]
        nrows_win = int(n_profiles / 2)
        ncols_win = int(n_profiles / nrows_win)
        fig, ax_list = plt.subplots(nrows=nrows_win, ncols=ncols_win, figsize=figsize,
                                    sharex='all', sharey='all', squeeze=True)

        for i, profile in enumerate(profiles):
            title, x_val, y_val, pts_list = profile

            ax = ax_list[int(i / 2), i % 2]
            ax.plot(x_val, y_val)
            ax.set_title(title)
            for pts in pts_list:
                x_pts, y_pts, colour = pts
                # n_on = len(on_rows)
                # mark_y = [cut_level] * n_on
                ax.plot(x_pts, y_pts, linestyle='none', marker='x', color=colour)
                # ax.plot(off_rows, mark_y, linestyle='none', marker='+', color='red')

        plt.show()
