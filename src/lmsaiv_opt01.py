#!/usr/bin/env python
"""
Decorators for use in all LMS projects.  Currently just includes @debug

@author: Alistair Glasse

Update:
"""
import astropy.units as u
import numpy as np
from astropy.convolution import convolve, Box1DKernel

from lms_mosaic import Mosaic
from lmssim_model import Model
from lms_globals import Globals
from lmsaiv_opt_tools import OptTools
from lmsaiv_plot import Plot
from lms_filer import Filer


class Opt01:

    def __init__(self):
        _ = Model()
        return

    @staticmethod
    def fov(title, as_built, **kwargs):
        """ Field of view calculation using flood illuminated continuum spectral images.  Populates the slice bounds
        map in the AsBuilt object
        """
        Opt01._analyse_darks()
        for opticon in [Globals.nominal, Globals.extended]:
            Opt01._find_fov(opticon, as_built)
        print('Done')
        return as_built

    @staticmethod
    def _analyse_darks():
        inc_tags = ['lms_opt_01', '_dark']
        darks = Filer.read_mosaic_list(inc_tags)
        OptTools.dark_stats(darks)
        if Globals.is_debug('low'):
            title = 'Dark'
            Plot.mosaic(darks[0], title=title)
            Plot.histograms(darks[0])
        return

    @staticmethod
    def _find_fov(opticon, as_built):
        opticon_tag = opticon[0:3]
        mosaics = Filer.read_mosaic_list(['lms_opt_01', 'flat_lamp', opticon_tag])
        # Coadd all flood images to allow full coverage (using multiple LMS configurations)
        flood = None
        for mosaic in mosaics:
            Plot.mosaic(mosaic, title=mosaic[0])
            if flood is None:
                flood = Mosaic.copy_mosaic(mosaic, clear_data=False, copy_name='')
                continue
            flood = Mosaic.sum_mosaics(flood, mosaic)

        if Globals.is_debug('low'):
            Plot.mosaic(flood, title='Coadded flood illumination')
        profiles = Opt01._find_slices(flood, smooth=3, snr_cut=5)
        slice_map = Opt01._make_slice_map(profiles, flood)
        Plot.mosaic(slice_map, title='Slice Map', cmap='hsv', mask=(0.0, 'black'))
        Opt01._calculate_fov(slice_map)
        Opt01._find_rrf(flood, slice_map)
        as_built['slice_map_' + opticon] = slice_map
        return as_built

    @staticmethod
    def _find_rrf(flood, slice_map):
        # Generate relative response tuple.
        cols = np.arange(0, 4096, 1)
        rrf = OptTools.copy_mosaic(slice_map, copy_name='rel_res_function')
        rrf_name, rrf_primary_header, rrf_hdus = rrf

        name, primary_hdr, hdus = flood
        wave_mosaic_cen = primary_hdr['HIERARCH ESO INS WLEN CEN'] * u.micron
        _, _, slice_map_hdus = slice_map
        for i in range(0, 4):
            slice_map_data = slice_map_hdus[i].data
            slice_mask = np.where(slice_map_data > 0., 1., 0.)
            # Very approximate dispersion...!
            hdr = hdus[i].header
            flood_image = hdus[i].data
            x_det_cen = float(hdr['X_CEN']) * u.mm
            n_det_rows, n_det_cols = flood_image.shape
            pix_size = float(hdr['HIERARCH AIT PIXEL_PITCH']) * u.mm
            c_det_cen = x_det_cen / pix_size
            c_det_org = c_det_cen - n_det_cols / 2
            disp = .08 * u.micron / (2. * n_det_cols)
            waves = wave_mosaic_cen + disp * (c_det_org + cols)
            flux = Model.black_body(waves, tbb=1000.)
            rrf_image = rrf_hdus[i].data
            for row in range(0, n_det_rows):
                idx = np.argwhere(slice_mask[row] > 0.)
                rrf_image[row, idx] = flood_image[row, idx] / flux[idx]
            rrf_hdus[i].data = rrf_image
        Plot.mosaic(rrf, title='Rel Response Function', cmap='grey', mask=(0.0, 'black'))
        return

    @staticmethod
    def _find_slices(mosaic, smooth=None, snr_cut=5):
        """ Calculate fov and return dictionary of slice_bounds and profiles used to calculate them.
        """
        file_name, hdr, hdus = mosaic
        opticon = hdr['HIERARCH AIT OPTICON']
        # Slice order from low to high detector number and low to high row number
        # opticon: {det_nos_12: (spifu_start, spifu_end, slice_start, slice_end),
        #           det_nos_34: (spifu_start, spifu_end, slice_start, slice_end)
        slice_order = {Globals.nominal: {'12': (0, 0, 15, 28), '34': (0, 0, 1, 14)},
                       Globals.extended: {'12': (1, 3, 11, 13), '34': (4, 6, 11, 13)}
                       }
        cut = 0.5       # Fraction of bright signal defining cut level
        print()
        print("File = {:s}".format(file_name))
        print("Identifying slices from along column profiles of flood illuminated images ")
        fmt = "Design fov, alpha pixel x slice width = {} x {}"
        print(fmt.format(Globals.alpha_pix, Globals.beta_slice))

        # Approximate slice image dimensions
        slice_width = 140
        slice_hw = int(0.5 * slice_width)
        gap = 15
        gap_hw = int(0.5 * gap)
        spifu_gap = 200

        slice_coords = {'det_nos': [], 'slice_nos': [], 'spifu_nos': [],
                        'col_mins': [], 'col_maxs': [], 'row_mins': [], 'row_maxs': []}

        profile_column_list = {1: [600, 800, 1000, 1200], 2: [300, 1400, 1700, 2000],
                               3: [600, 700, 800, 1200], 4: [300, 1600, 1800, 2000]}
        profiles = []
        for hdu in hdus:
            det_no = int(hdu.header['ID'])
            det_tag = {1: '12', 2: '12', 3: '34', 4: '34'}[det_no]
            det_slice_order = slice_order[opticon]
            spifu_start, spifu_end, slice_start, slice_end = det_slice_order[det_tag]

            img = hdu.data
            profile_columns = profile_column_list[det_no]
            for profile_column in profile_columns:
                spifu_no = spifu_start
                slice_no = slice_start
                pc1, pc2 = profile_column - 4, profile_column + 5
                if Globals.is_debug('high'):
                    print('Opt01._find_slices, '
                          'det_no= ', det_no, 'spifu_no= ', spifu_no, 'slice_no= ',
                          slice_no, 'col= ', profile_column)
                signal = np.mean(img[:, pc1:pc2], axis=1)
                if smooth is not None:
                    boxcar = Box1DKernel(smooth)
                    signal = np.convolve(signal, boxcar, mode='same')
                original_signal = np.array(signal)
                bgd_noise_level = np.std(signal[0:70])
                row_lo = 0
                pts = []
                more_rows = True
                while more_rows:
                    try:
                        row_lo += np.argwhere(signal[row_lo:] > snr_cut * bgd_noise_level)[0][0]
                    except IndexError:
                        fmt = 'Profile detection failed at det_no= {:d}, col= {:d}, row= {:d}'
                        print(fmt.format(det_no, profile_column, row_lo))
                    row_bright = row_lo + np.argmax(signal[row_lo:row_lo + slice_hw])
                    y_bright = signal[row_bright]                 # Typical peak signal in slice
                    y_cut = cut * y_bright

                    row_lo += np.argwhere(signal[row_lo:] > cut * y_bright)[0][0]      # Row after 50 % point
                    ya, yb = signal[row_lo-1], signal[row_lo]
                    dr = (yb - ya) / y_cut
                    rlo = row_lo + dr - 1                           # Add pixel fraction for cut level.

                    row_hi = row_bright + np.argwhere(signal[row_bright:] < y_cut)[0][0]
                    ya, yb = signal[row_hi - 1], signal[row_hi]
                    dr = (yb - ya) / y_cut
                    rhi = row_hi - dr - 1
                    if Globals.is_debug('high'):
                        print("- {:5.2f}, {:5d}, {:5.2f}, {:5.3f}".format(rlo, row_bright, rhi, y_cut))
                    slice_coords['det_nos'].append(det_no)
                    slice_coords['slice_nos'].append(slice_no)
                    slice_coords['spifu_nos'].append(spifu_no)
                    slice_coords['col_mins'].append(pc1)
                    slice_coords['col_maxs'].append(pc2)
                    slice_coords['row_mins'].append(rlo)
                    slice_coords['row_maxs'].append(rhi)

                    pts.append((rlo, y_cut))
                    pts.append((rhi, y_cut))

                    # Remove slice from profile data
                    # signal[row_lo - gap_hw: row_hi + gap_hw] = 0.
                    signal[0: row_hi + gap_hw] = 0.

                    slice_no += 1
                    if slice_no > slice_end:        # this should only be true in extended mode.
                        signal[row_lo - gap_hw: row_hi + spifu_gap] = 0.
                        slice_no = slice_start
                        spifu_no += 1
                        if spifu_no > spifu_end:
                            more_rows = False
                label = "col={}".format(profile_column)
                profiles.append((label, det_no, profile_column, original_signal, pts))


        # Convert lists to numpy arrays
        for key in slice_coords:
            slice_coords[key] = np.array(slice_coords[key])
        if Globals.is_debug('low'):
            Plot.profiles(profiles)
        return slice_coords

    @staticmethod
    def _make_slice_map(slice_coords, mosaic):
        """ Create slice map, which is a fits HDU detector mosaic image where each pixel takes the value of
        its slice number N, such that N = slice_no + 100 x spifu_no
        """
        slice_map = OptTools.copy_mosaic(mosaic, clear_data=True, copy_name='slice_map')
        slice_map_name, slice_map_hdr, slice_map_hdus = slice_map
        opticon = slice_map_hdr['HIERARCH AIT OPTICON']

        for hdu in slice_map_hdus:
            det_no = int(hdu.header['ID'])
            det_no_idxs = slice_coords['det_nos'] == det_no
            uni_spifu_nos = np.unique(slice_coords['spifu_nos'][det_no_idxs])
            for spifu_no in uni_spifu_nos:
                spifu_no_idxs = np.logical_and(slice_coords['spifu_nos'] == spifu_no, det_no_idxs)
                uni_slice_nos = np.unique(slice_coords['slice_nos'])
                for slice_no in uni_slice_nos:
                    idxs = np.logical_and(slice_coords['slice_nos'] == slice_no, spifu_no_idxs)
                    row_mins = np.array(slice_coords['row_mins'])[idxs]
                    if len(row_mins) < 1:       # Catch cases where the slice is not on the detector
                        continue
                    row_maxs = np.array(slice_coords['row_maxs'])[idxs]
                    col_mins = np.array(slice_coords['col_mins'])[idxs]
                    col_maxs = np.array(slice_coords['col_maxs'])[idxs]
                    cols = 0.5 * (col_mins + col_maxs)
                    row_min_fit = np.polyfit(cols, row_mins, 2)
                    row_max_fit = np.polyfit(cols, row_maxs, 2)

                    nr, nc = hdu.data.shape
                    cs = np.arange(0, nc, 1)
                    r1s = np.rint(np.polyval(row_min_fit, cs))
                    r2s = np.rint(np.polyval(row_max_fit, cs))
                    for c, r1, r2 in zip(cs, r1s, r2s):
                        hdu.data[int(r1):int(r2), int(c)] = int(slice_no + 100 * spifu_no)
        return slice_map

    @staticmethod
    def _calculate_fov(slice_map):
        _, pri_hdu, hdu_list = slice_map
        opticon = pri_hdu['HIERARCH AIT OPTICON']
        n_rows, n_cols = Globals.det_format
        n_illum = 0
        for hdu in hdu_list:
            n_illum += np.count_nonzero(hdu.data)

        n_slices = 28 if opticon == Globals.nominal else 3
        n_spifus = 1 if opticon == Globals.nominal else 6
        beta_fov = Globals.beta_slice * n_slices
        n_alphas = n_illum / n_slices / n_cols / n_spifus / 2
        alpha_fov = n_alphas * Globals.alpha_pix
        print("Field of view = {:9.2f} x {:9.2f}".format(alpha_fov, beta_fov))
        return

    @staticmethod
    def flood_stats(mosaic):
        """ Calculate fov and return dictionary of slice_bounds and profiles used to calculate them.
        """
        file_name, hdr, hdus = mosaic
        opticon = hdr['HIERARCH ESO INS MODE']

        # Set up slice map object to hold slice images
        slice_map = OptTools.copy_mosaic(mosaic, clear_data=True, copy_name='slice_map')

        u.arcsec2 = u.arcsec * u.arcsec

        dark_pctile = 10.
        bright_pctile = 90.         # Choose the bright pixel limit to avoid hot pixels.
        alpha_cut = 0.5
        print()
        print("File = {:s}".format(file_name))
        fmt = "Dark pixels are defined as those < {:.0f}th percentile signal level"
        print(fmt.format(dark_pctile))
        fmt = "Illuminated pixels defined as those brighter than the {:.0f}th percentile signal level"
        print(fmt.format(bright_pctile))
        fmt = "alpha extent of each slice defined as distance between {:3.2f} of bright level"
        print(fmt.format(alpha_cut))
        fmt = "Illuminated pixel x slice fov = {} x {} mas"
        print(fmt.format(Globals.alpha_pix, Globals.beta_slice))
        profile_cols = {1: [600, 800, 1000, 1200], 3: [600, 700, 800, 1200],
                        2: [1000, 1800, 1900, 2000], 4: [1000, 1800, 1900, 2000]}

        print()
        fmt = "{:>10s},{:>10s},{:>10s},{:>10s}"
        print(fmt.format('Detector', 'dark',  'bright',    'illum.'))
        print(fmt.format('        ', 'level', 'level',        'fov'))
        print(fmt.format('        ', 'DN',       'DN',  '[sq_asec]'))
        fmt = "{:>10d},{:>10.2e},{:>10.2e},{:>10.3f}"

        # Separate slices by finding cuts in d_signal / d_row
        n_slices = 28 if '_nom_' in file_name else 3
        n_spifus = 0 if '_nom_' in file_name else 6

        profiles = []
        alpha_det = [0.]*4
        for hdu in hdus:
            det_no = int(hdu.header['ID'])
            img = np.array(hdu.data)            # Copy the image data
            det_idx = det_no - 1
            slice_coords = {'det_no': det_no, 'slice_nos': [], 'cols': [], 'row_mins': [], 'row_maxs': []}

            slice_no = 1

            n_profiles = len(profile_cols[det_no])
            dark_level, bright_level = 0., 0.           # Average signal cut levels for this detector
            for pr_col in profile_cols[det_no]:
                pr_col_hw = 2                           # Co-add 2 x pr_col_hw + 1 centred on pr_col.
                col1, col2 = pr_col - pr_col_hw, pr_col + pr_col_hw
                y_signal = np.nanmean(img[:, col1:col2+1], axis=1)
                y_noise = np.nanstd(img[:, col1:col2+1], axis=1)

                # Find the rows bounding the top 10 %ile, then replace them with the bottom 5 %ile value.
                cut_level = 0.01 * np.amax(y_signal)
                row_off = 0
                spifu_no = 0
                slice_count = 0
                on_rows = []
                off_rows = []
                while True:
                    # Find next bright pixel
                    on_indices = np.argwhere(y_signal[row_off:] > cut_level)
                    if len(on_indices) < 1:       # No more bright pixels found.
                        break
                    row_on = row_off + on_indices[0][0]
                    r_on = np.interp(cut_level, [y_signal[row_on-1], y_signal[row_on]], [row_on-1, row_on])
                    off_indices = np.argwhere(y_signal[row_on:] < cut_level)
                    row_off = row_on + off_indices[0][0]
                    slice_coords['cols'].append(pr_col)
                    slice_coords['row_mins'].append(row_on)
                    slice_coords['row_maxs'].append(row_off)
                    slice_coords['slice_nos'].append(slice_no)
                    r_off = np.interp(cut_level, [y_signal[row_off], y_signal[row_off-1]], [row_off, row_off-1])

                    alpha_slice = (row_off - r_on) * Globals.alpha_pix
                    alpha_det[det_idx] += alpha_slice
                    on_rows.append(r_on)
                    off_rows.append(r_off)

                    if n_spifus == 0:
                        slice_no += 1
                    else:                       # MSA set to 'extended'
                        slice_count += 1
                        slice_no += 1
                        if slice_count % n_slices == 0:
                            slice_no -= n_slices
                            if spifu_no != 0:
                                spifu_no += 1
                x_indices = np.arange(0, len(y_signal))
                title = "det {:d}, cols {:d}-{:d}".format(det_no, col1, col2)
                on_points = on_rows, [cut_level] * len(on_rows), 'green'
                off_points = off_rows, [cut_level] * len(off_rows), 'blue'
                profile = title, x_indices, y_signal, [on_points, off_points]
                profiles.append(profile)
                # Calculate total length of illuminated slices.
                illum_rows_sum = 0.
                for row_on, row_off in zip(on_rows, off_rows):
                    illum_rows = row_off - row_on
                    illum_rows_sum += illum_rows

            # Create the slice map (and calculate the intra-slice gap) from the table of slice bounds.
            poly_degree = 2 if n_profiles > 3 else n_profiles - 1

            slice_nos = np.array(slice_coords['slice_nos'], dtype='int64')
            unique_slice_nos = np.unique(slice_nos)

            slice_map_name, slice_map_hdr, slice_map_hdus = slice_map
            slice_map_hdu = slice_map_hdus[det_idx]
            for slice_no in unique_slice_nos:
                idx = np.where(slice_no == slice_nos)[0]
                cols = np.array(slice_coords['cols'])[idx]
                row_mins = np.array(slice_coords['row_mins'])[idx]
                row_maxs = np.array(slice_coords['row_maxs'])[idx]
                row_min_fit = np.polyfit(cols, row_mins, poly_degree)
                row_max_fit = np.polyfit(cols, row_maxs, poly_degree)
                nr, nc = slice_map_hdus[det_idx].data.shape
                cs = np.arange(0, nc, 1)
                r1s = np.rint(np.polyval(row_min_fit, cs))
                r2s = np.rint(np.polyval(row_max_fit, cs))
                for c, r1, r2 in zip(cs, r1s, r2s):
                    slice_map_hdu.data[int(r1):int(r2), int(c)] = slice_no + 100 * spifu_no
            alpha_det[det_idx] /= n_profiles
            det_fov = alpha_det[det_idx] * Globals.beta_slice
            fmt = "{:>10d},{:>10.1f},{:>10.1f},{:>10.3f}"
            print(fmt.format(det_no, dark_level, bright_level, det_fov.to(u.arcsec2)))

        alpha_02 = alpha_det[0] + alpha_det[2]
        alpha_13 = alpha_det[1] + alpha_det[3]
        alpha_ave = 0.5 * (alpha_02 + alpha_13)
        fov = alpha_ave * Globals.beta_slice
        print()
        print("Total field of view = {:5.3f} (cf METIS-3667, shall be > 0.500 arcsec2)".format(fov.to(u.arcsec2)))
        alpha_ext, beta_ext = alpha_ave / n_slices, Globals.beta_slice * n_slices
        aspect_ratio = alpha_ext / beta_ext
        print("Aspect ratio (alpha/beta) = {:5.3f}:1 (cf METIS-3667, shall be 1:1 < ar < 2:1)".format(aspect_ratio))
        return slice_map, profiles

    @staticmethod
    def print_inter_slice(slice_map):
        """ Brute force calculation of inter slice gap from slice map """
        _, _, hdu_list = slice_map
        fmt = "{:>12s},{:>12s},{:>12s},{:>12s},"
        print(fmt.format('Detector', 'Slice No.', 'Slice No.', 'Minimum'))
        print(fmt.format('No.', 'A', 'B', 'Gap'))
        fmt = "{:>12d},{:>12d},{:>12d},{:>12d},"
        for hdu in hdu_list:
            header, img = hdu.header, hdu.data
            det_no = int(header['ID'])
            # Sufficient to find slice numbers by searching along the central column (1024)
            values = np.unique(img[:, 1024])
            slice_nos = values[values > 0]
            for slice_no in slice_nos[1:]:
                gap_list = []
                for c in range(0, 2048):
                    cut = img[:, c]
                    row_a_top = np.argwhere(cut == slice_no - 1)[-1]
                    row_b_bot = np.argwhere(cut == slice_no)[0]      # Get indices of non-slice pixels.
                    gap_list.append(row_b_bot - row_a_top)
                gaps = np.array(gap_list)
                print(fmt.format(det_no, int(slice_no - 1), int(slice_no), gaps.min()))

        print("Inter slice gap calculation")
        return



    @staticmethod
    def _parse_abo(file_name):
        """ Extract alpha, beta and observation number from a fits file name.  Used in lms_opt_01_t2
        """
        signed = {'_a': True, '_b': True, '_o': False}
        ip = file_name.find('_grid') + 5        # Get start position of wavelength, alpha and beta sub strings
        abo = []
        for tag in signed:
            ip = file_name.find(tag, ip) + len(tag)
            sign = 1.0
            if signed[tag]:
                sign = 1. if file_name[ip] == 'p' else -1.
                ip += 1
            mag = float(file_name[ip: ip + 3])
            val = sign * mag
            abo.append(val)
        return tuple(abo)

