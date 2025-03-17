import numpy as np
from astropy import units as u
from astropy.table import QTable
from lms_globals import Globals


class OptTools:


    def __init__(self):
        return

    @staticmethod
    def dark_stats(mosaics):
        for mosaic in mosaics:
            file_name, hdr, data = mosaic
            print()
            print("File = {:s}".format(file_name))
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
        """ Calculate fov and return dictionary of slice_bounds and profiles used to calculate them.
        """
        file_name, hdr, data = mosaic
        u.arcsec2 = u.arcsec * u.arcsec

        dark_pctile = 10.
        bright_pctile = 90.
        alpha_cut = 0.5
        print()
        print("File = {:s}".format(file_name))
        fmt = "Dark pixels defined as those < {:.0f}th percentile signal level"
        print(fmt.format(dark_pctile))
        fmt = "Illuminated pixels defined as those brighter than the {:.0f}th percentile signal level"
        print(fmt.format(bright_pctile))
        fmt = "alpha extent of each slice defined as distance between {:3.2f} of bright level"
        print(fmt.format(alpha_cut))
        fmt = "Illuminated pixel x slice fov = {} x {} mas"
        print(fmt.format(Globals.alpha_pix, Globals.beta_slice))
        profile_col = {1: 700, 3: 700, 2: 2000, 4: 2000}

        print()
        fmt = "{:>10s},{:>10s},{:>10s},{:>10s}"
        print(fmt.format('Detector', 'dark',  'bright',    'illum.'))
        print(fmt.format('        ', 'level', 'level',        'fov'))
        print(fmt.format('        ', 'DN',       'DN',  '[sq_asec]'))
        fmt = "{:>10d},{:>10.2e},{:>10.2e},{:>10.3f}"

        # Separate slice images by faint level crossings of column averaged profile
        n_slices = 28 if '_nom_' in file_name else 3
        n_spifus = 0 if '_nom_' in file_name else 6

        sb_names = ['det_no', 'slice_no', 'spifu_no',
                    'det_col', 'det_row_min', 'det_row_max']
        slice_bounds = QTable(names=sb_names, dtype=['i4', 'i4', 'i4', 'f8', 'f8', 'f8'])
        profiles = []
        alpha_det = [0.]*4
        for i, image in enumerate(data):
            det_no = i + 1
            pr_col = profile_col[det_no]
            pr_col_hw = 5
            col1, col2 = pr_col - pr_col_hw, pr_col + pr_col_hw
            col_cen = int((col1 + col2) / 2)
            fmt = "Analysing profile at column {:d} averaged over columns {:d} to {:d}"
            print(fmt.format(col_cen, col1, col2))

            slice_no = 1 if det_no in [3, 4] else 14  # Starting slice number (bottom row)
            spifu_start = 0
            if n_spifus > 0:
                slice_no = 12
                spifu_start = 1 if det_no in [3, 4] else 4
            spifu_no = spifu_start
            y_vals = np.nanmean(image[:, col1:col2], axis=1)
            faint_level = np.nanpercentile(y_vals, dark_pctile)
            bright_level = np.nanpercentile(y_vals, bright_pctile)
            on_rows, off_rows = [], []
            row_off = 0
            cut_level = alpha_cut * bright_level
            slice_count = 0
            while True:
                # Find next bright pixel
                print(slice_count, slice_no, spifu_no)

                on_indices = np.argwhere(y_vals[row_off:] > 1. * cut_level)
                if len(on_indices) < 1:       # No more bright pixels found.
                    break
                row_on = row_off + on_indices[0][0]
                r_on = np.interp(cut_level, [y_vals[row_on-1], y_vals[row_on]], [row_on-1, row_on])
                off_indices = np.argwhere(y_vals[row_on:] < 1. * cut_level)
                row_off = row_on + off_indices[0][0]
                r_off = np.interp(cut_level, [y_vals[row_off], y_vals[row_off-1]], [row_off, row_off-1])
                sb_row = [det_no, slice_no, spifu_no, col_cen, r_on, r_off]
                slice_bounds.add_row(sb_row)

                alpha_slice = (row_off - r_on) * Globals.alpha_pix
                alpha_det[i] += alpha_slice
                on_rows.append(r_on)
                off_rows.append(r_off)

                if n_spifus == 0:
                    slice_no += 1
                else:                       # MSA set to 'extended'
                    slice_count += 1
                    if slice_count % n_slices == 0:
                        slice_no += 1
                    n = int(n_spifus / 2)
                    if slice_count % n == 0:
                        spifu_no = spifu_start + slice_count % n
                    else:
                        spifu_no += 1


            x_indices = np.arange(0, len(y_vals))
            title = "det {:d}, cols {:d}-{:d}".format(i+1, col1, col2)
            on_points = on_rows, [cut_level] * len(on_rows), 'green'
            off_points = off_rows, [cut_level] * len(off_rows), 'blue'
            profile = title, x_indices, y_vals, [on_points, off_points]
            profiles.append(profile)
            # Calculate total length of illuminated slices.
            illum_rows_sum = 0.
            for row_on, row_off in zip(on_rows, off_rows):
                illum_rows = row_off - row_on
                illum_rows_sum += illum_rows

            det_fov = alpha_det[i] * Globals.beta_slice  # Pixel FOV in arcsec^2
            # print(fmt.format(i, faint_level, bright_level, det_fov.to(u.arcsec2).value))

        alpha_02 = alpha_det[0] + alpha_det[2]
        alpha_13 = alpha_det[1] + alpha_det[3]
        alpha_ave = 0.5 * (alpha_02 + alpha_13)
        fov = alpha_ave * Globals.beta_slice
        print()
        print("Total field of view = {:5.3f}".format(fov.to(u.arcsec2)))
        alpha_ext, beta_ext = alpha_ave / n_slices, Globals.beta_slice * n_slices
        aspect_ratio = alpha_ext / beta_ext
        print("Aspect ratio (alpha/beta) = {:5.3f}:1 ".format(aspect_ratio))

        return slice_bounds, profiles

    @staticmethod
    def extract_alpha_trace(mosaic):
        trace = None
        name, hdr, data = mosaic
        alpha_det = [0.]*4
        for i, image in enumerate(data):
            col = 400
            profile = image[:, col]
            nob = 1
            # faint_level = np.nanpercentile(profile, dark_pctile)
            # bright_level = np.nanpercentile(profile, bright_pctile)
            # on_rows, off_rows, slice_bounds = [], [], []
            # row_off = 0
        return trace

