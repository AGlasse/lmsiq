import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from lms_globals import Globals


class OptTools:


    def __init__(self):
        return

    @staticmethod
    def plot_mosaic(mosaic, **kwargs):
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
        # ax.cax.toggle_label(True)
        plt.show()
        return

    @staticmethod
    def dark_stats(mosaic):
        file_name, hdr, data = mosaic
        print("{:>10s},{:>10s},{:>10s}".format('Detector', 'median', 'stdev'))

        for i, image in enumerate(data):
            median = np.median(image)
            stdev = np.std(image)
            fmt = "{:10d},{:10.3f},{:10.3f}"
            text = fmt.format(i + 1, median, stdev)
            print(text)
        return

    @staticmethod
    def flood_stats(mosaic):
        file_name, hdr, data = mosaic

        n_slices = 28 if 'nom' in file_name else 3

        dark_pctile = 10.
        bright_pctile = 90.
        alpha_cut = 0.5
        print()
        fmt = "Dark pixels defined as those < {:.0f}th percentile signal level"
        print(fmt.format(dark_pctile))
        fmt = "Illuminated pixels defined as those brighter than the {:.0f}th percentile signal level"
        print(fmt.format(bright_pctile))
        fmt = "alpha extent of each slice defined as distance between {:3.2f} of bright level"
        print(fmt.format(alpha_cut))
        fmt = "Illuminated pixel x slice fov = {} x {} mas"
        print(fmt.format(Globals.alpha_mas_pix, Globals.beta_mas_slice))
        fmt = "Analysing profile averaged over columns {:d} to {:d}"
        col1, col2 = 500, 510
        print(fmt.format(col1, col2))

        print()
        fmt = "{:>10s},{:>10s},{:>10s},{:>10s}"
        print(fmt.format('Detector', 'dark',  'bright',    'illum.'))
        print(fmt.format('        ', 'level', 'level',        'fov'))
        print(fmt.format('        ', 'DN',       'DN',  '[sq_asec]'))
        fmt = "{:>10d},{:>10.2e},{:>10.2e},{:>10.3f}"

        figsize = [8, 8]
        fig, ax_list = plt.subplots(2, 2, figsize=figsize,
                                    sharex='all', sharey='all', squeeze=True)

        # Separate slice images by faint level crossings of column averaged profile
        alpha_det = [0.]*4
        for i, image in enumerate(data):
            profile = np.nanmean(image[:, col1:col2], axis=1)
            faint_level = np.nanpercentile(profile, dark_pctile)
            bright_level = np.nanpercentile(profile, bright_pctile)
            on_rows, off_rows, slice_bounds = [], [], []
            row_off = 0
            cut_level = alpha_cut * bright_level
            while True:
                # Find next bright pixel
                on_indices = np.argwhere(profile[row_off:] > 1. * cut_level)
                if len(on_indices) < 1:       # No more bright pixels found.
                    break
                row_on = row_off + on_indices[0][0]
                r_on = np.interp(cut_level, [profile[row_on-1], profile[row_on]], [row_on-1, row_on])
                off_indices = np.argwhere(profile[row_on:] < 1. * cut_level)
                row_off = row_on + off_indices[0][0]
                r_off = np.interp(cut_level, [profile[row_off], profile[row_off-1]], [row_off, row_off-1])
                alpha_slice = (row_off - r_on) * Globals.alpha_mas_pix
                alpha_det[i] += alpha_slice
                # print('alpha-slice = ', alpha_slice)
                slice_bound = int(.5 * (row_on + row_off))
                slice_bounds.append(slice_bound)
                on_rows.append(r_on)
                off_rows.append(r_off)
            ax = ax_list[int(i / 2), i % 2]
            x_indices = np.arange(0, len(profile))
            ax.plot(x_indices, profile)
            ax.set_title("det {:d}, cols {:d}-{:d}".format(i+1, col1, col2))
            n_on = len(on_rows)
            mark_y = [cut_level]*n_on
            ax.plot(on_rows, mark_y, linestyle='none', marker='x', color='green')
            ax.plot(off_rows, mark_y, linestyle='none', marker='+', color='red')

            # Calculate total length of illuminated slices..
            illum_rows_sum = 0.
            for row_on, row_off in zip(on_rows, off_rows):
                illum_rows = row_off - row_on
                illum_rows_sum += illum_rows

            det_fov = alpha_det[i] * Globals.beta_mas_slice / 1.E+6  # Pixel FOV in arcsec^2
            print(fmt.format(i, faint_level, bright_level, det_fov))
        alpha_02 = alpha_det[0] + alpha_det[2]
        alpha_13 = alpha_det[1] + alpha_det[3]
        alpha_ave = 0.5 * (alpha_02 + alpha_13)
        fov = alpha_ave * Globals.beta_mas_slice / 1.E+6
        print()
        print("Total field of view = {:5.3f} sq. arcsec".format(fov))
        alpha_ext, beta_ext = alpha_ave / n_slices, Globals.beta_mas_slice * n_slices
        aspect_ratio = alpha_ext / beta_ext
        print("Aspect ratio (alpha/beta) = {:5.3f}:1 ".format(aspect_ratio))

        plt.show()
        return

    def plot_profile(profile, mark_x_list=[], figsize=[8, 8]):
        fig, ax = plt.subplots(1, 1, figsize=figsize,
                               sharex=True, sharey=True, squeeze=True)
        x_indices = np.arange(0, len(profile))
        ax.plot(x_indices, profile)
        for mark_x in mark_x_list:
            mark_y = profile[mark_x]
            ax.plot(mark_x, mark_y, linestyle='none', marker='x')
        plt.show()
        return
