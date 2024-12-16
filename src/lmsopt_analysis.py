import numpy as np
import matplotlib.pyplot as plt
from lms_globals import Globals
from lms_filer import Filer


class OptAnalysis:
    common_path = '../data/test_sim/lms_opt_01_t1_'

    def __init__(self):
        return

    def run(self, test_name):
        title_tag = test_name.lower()
        data_folder = '../data/test_sim/'
        file_list = Filer.get_file_list(data_folder, inc_tags=[title_tag, '.fits'])

        print('Data files found for test '.format(test_name))
        for file in file_list:
            print(file)
        match test_name:
            case 'lms_opt_01_t1':
                self.lms_opt_01_t1()
            case 'lms_opt_01_t2':
                self.lms_opt_01_t2()
            case _:
                return "Test analysis method not yet supported"

    @staticmethod
    def read_mosaic(path):
        return Filer.read_fits(path, data_exts=[1, 2, 3, 4])

    @staticmethod
    def plot_profile(profile, mark_x_list=[]):
        figsize = [8, 8]
        fig, ax = plt.subplots(1, 1, figsize=figsize,
                               sharex=True, sharey=True, squeeze=True)
        x_indices = np.arange(0, len(profile))
        ax.plot(x_indices, profile)
        for mark_x in mark_x_list:
            mark_y = profile[mark_x]
            ax.plot(mark_x, mark_y, linestyle='none', marker='x')
        plt.show()
        return

    @staticmethod
    def plot_mosaic(mosaic, **kwargs):
        nrows, ncols = 2, 2
        figsize = [8, 8]
        fig, ax_list = plt.subplots(nrows, ncols, figsize=figsize,
                                    sharex=True, sharey=True, squeeze=True)
        xmin, xmax = 0, mosaic[0].shape[0]
        ymin, ymax = 0, mosaic[0].shape[1]
        vmin = kwargs.get('vmin', np.nanmin(mosaic))
        vmax = kwargs.get('vmax', np.nanmax(mosaic))
        for det_idx in range(0, 4):
            mos_row = det_idx // 2
            mos_col = det_idx % 2
            ax = ax_list[mos_row, mos_col]
            ax.set_xlim(xmin-1, xmax+1)
            ax.set_ylim(ymin-1, ymax+1)
            ax.set_aspect('equal', 'box')
            image = mosaic[det_idx]
            ax.imshow(image, extent=(xmin-0.5, xmax+0.5, ymin-0.5, ymax+0.5),
                      interpolation='nearest', cmap='hot', vmin=vmin, vmax=vmax, origin='lower')
        plt.show()
        return

    @staticmethod
    def dark_stats(name):
        print(name)
        header, dark = OptAnalysis.read_mosaic(OptAnalysis.common_path + name + '.fits')
        print("{:>10s},{:>10s},{:>10s}".format('Detector', 'median', 'stdev'))
        for i, image in enumerate(dark):
            median = np.median(image)
            stdev = np.std(image)
            fmt = "{:10d},{:10.3f},{:10.3f}"
            text = fmt.format(i + 1, median, stdev)
            print(text)
        return

    @staticmethod
    def flood_stats(name):
        header, flood = OptAnalysis.read_mosaic(OptAnalysis.common_path + name + '.fits')
        dark_pctile = 10.
        bright_pctile = 90.
        print()
        fmt = "Dark pixels defined as those < {:.0f}th percentile signal level"
        print(fmt.format(dark_pctile))
        fmt = "Illuminated pixels defined as those brighter than the {:.0f}th percentile signal level"
        print(fmt.format(bright_pctile))
        fmt = "Illuminated pixel x slice fov = {} x {} mas"
        print(fmt.format(Globals.alpha_mas_pix, Globals.beta_mas_pix))
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
                               sharex=True, sharey=True, squeeze=True)

        for i, image in enumerate(flood):
            # Separate slice images by faint level crossings of column averaged profile
            profile = np.nanmean(image[:, col1:col2], axis=1)
            faint_level = np.nanpercentile(profile, dark_pctile)
            bright_level = np.nanpercentile(profile, bright_pctile)
            on_rows, off_rows, slice_bounds = [], [], []
            off_row = 0
            cut_on = .5 * bright_level
            while True:
                on_index = np.argwhere(profile[off_row:] > 1. * cut_on)
                if len(on_index) < 1:
                    slice_bound = int(0.5 * (2047 + off_row))
                    slice_bounds.append(slice_bound)
                    break
                on_row = off_row + on_index[0][0]
                slice_bound = int(.5 * (on_row + off_row))
                slice_bounds.append(slice_bound)
                on_rows.append(on_row)
                off_row = on_row + np.argwhere(profile[on_row:] < 1. * cut_on)[0][0]
                off_rows.append(off_row)
            ax = ax_list[int(i / 2), i % 2]
            x_indices = np.arange(0, len(profile))
            ax.plot(x_indices, profile)
            ax.set_title("det {:d}, cols {:d}-{:d}".format(i+1, col1, col2))
            for mark_x in [on_rows, off_rows]:
                mark_y = profile[mark_x]
                ax.plot(mark_x, mark_y, linestyle='none', marker='x')

            # OptAnalysis.plot_profile(profile, mark_x_list=[on_rows, off_rows])

            # Calculate total length of illuminated slices..
            illum_rows_sum = 0.
            for on_row, off_row in zip(on_rows, off_rows):
                illum_rows = off_row - on_row
                illum_rows_sum += illum_rows

            alpha_sum = illum_rows_sum * Globals.alpha_mas_pix
            illum_fov = alpha_sum * Globals.beta_mas_pix / 1.E+6  # Pixel FOV in arcsec^2
            print(fmt.format(i, faint_level, bright_level, illum_fov))
        plt.show()
        OptAnalysis.plot_mosaic(flood)
        return

    def lms_opt_01_t1(self):
        self.dark_stats('nom_dark1')
        self.flood_stats('nom_flood')
        print('Done')
        return

    def lms_opt_01_t2(self):
        print('Done')
        return
