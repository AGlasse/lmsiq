#!/usr/bin/env python
"""

"""
import math
import time
import numpy as np
import scipy.signal
from astropy.io import fits
from lmsdist_util import Util
from lms_globals import Globals
from lms_filer import Filer
from lms_detector import Detector

tau_blaze_kernel = None     # Kernel blaze profile tau(x) where x = (wave / blaze_wave(eo) - 1)


def run():
    """ Run simulation,
    - using LMS entrance focal plane to detector focal plane transforms, calculated in lmsdist.py
    - using LMS PSFs, currently including the (significant) optical design aberrations, which will be updated using
      real wavefront error maps.
    - implementing calibration sources (WCU and sea-level sky) available during AIT in Leiden.
    Start by defining the observation.
    """
    analysis_type = 'distortion'

    coord_in = 'efp_x', 'efp_y', 'wavelength'
    coord_out = 'det_x', 'det_y'

    nominal = Globals.nominal
    nom_date_stamp = '20240109'
    nom_config = (analysis_type, nominal, nom_date_stamp,
                  'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
                  coord_in, coord_out)

    spifu = Globals.spifu
    spifu_date_stamp = '20231009'
    spifu_config = (analysis_type, spifu, spifu_date_stamp,
                    'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                    coord_in, coord_out)
    model_configurations = {nominal: nom_config, spifu: spifu_config}

    # Define a list of extended illumination sources.  These images will be convolved with the 'target slice' PSF.
    ext_src = {'wcu_bb': {'sed': 'bb', 'temperature': 1000.},               # 1000 K black body
               'wcu_pnh': {'sed': 'bb', 'temperature': 300.},
               'wcu_ls': {'sed': 'laser', 'flux': 1000., 'wavelength': 3390},
               'wcu_ll': {'sed': 'laser', 'flux': 1000., 'wavelength': 5240},
               'wcu_lt': {'sed': 'laser', 'flux': 1000., 'nlines': 10, 'wshort': 4590, 'wlong': 4850},
               'sky': {'sed': 'sky'}}    # Model sky emission spectrum
    # Define one or more (point-like) pinhole masks which will spatially filter the extended source.  The model
    # specified PSFs at +-4 slices from the target slice will be convolved with the 'pinhole' images.
    pnh_mask = {'cfo': {'efp_x': 0., 'efp_y': 0., 'ext': ext_src['sky']},       # On-axis pinhole in boresight
                'wcu': {'efp_x': 0., 'efp_y': 0., 'ext': ext_src['wcu_bb']},    # Steerable pinhole in WCU.
                'none': None}
    # ============ Configure observation here. ====================================================
    obs_cfg = {'id': 'lms-opt-01',
               'opticon': spifu,
               'wave_cen': 4.700,
               'ext': 'sky',
               'pnh': 'none'}
    # =============================================================================================
    obs_id = obs_cfg['id']
    opticon = obs_cfg['opticon']
    wave_cen = obs_cfg['wave_cen']
    ext_tag = obs_cfg['ext']
    pnh_tag = obs_cfg['pnh']
    fmt = "{:s}_{:s}_wav{:04d}_ext{:s}_pnh{:s}"
    obs_name = fmt.format(obs_id, opticon, int(wave_cen * 1000.), ext_tag, pnh_tag)
    print('Running simulation - ' + obs_name)

    model_config = model_configurations[opticon]
    filer = Filer(model_config)
    date_stamp = model_config[2]

    t_start = time.perf_counter()

    # Read header and data shape (only) in from the template.
    data_exts = [1, 2, 3, 4]
    hdr_primary, data_list = filer.read_fits('../data/sim_template.fits', data_exts=data_exts)
    det_shape = data_list[0].shape
    _, n_det_cols = det_shape

    # Model PSFs are available up to 4 slices away from the slice the target is centred on.
    zemax_psf_slice_range = [-4, 4]

    # Set up transmission terms
    tau_wcu_ap = 0.1                # WCU variable aperture flux setting (0.01 - 1.00?)
    tau = {'isphere': .1, 'wcu_app': tau_wcu_ap, 'wcu_pickoff': .5,
           'sky_pickoff': .1, 'pickoff_cfopnh': .5, 'cfopnh_lmsdet': .2}

    ext = ext_src[ext_tag]

    pinhole = pnh_mask[pnh_tag]
    if pinhole is not None:
        pinhole['slice_range'] = zemax_psf_slice_range
        slice_no_cen, _ = Util.efp_y_to_slice(pinhole['efp_y'])  # Pinhole is centred on this slice
        pinhole['slice_no_cen'] = slice_no_cen

    dit = 1.3  # Integration time in seconds.

    # Find EFP bounds
    efp_xmax = Globals.efp_x_fov_mm
    xh = efp_xmax / 2.
    wrange = 0.5        # Width of wavelength range to model (1 um to overfill mosaic in spifu mode)
    wbounds = [wave_cen - wrange/2., wave_cen + wrange/2.]
    srp_model = 200000
    dw_model = wbounds[0] / srp_model
    n_pts_model = int(wrange / dw_model) + 1
    # waves = np.linspace(wbounds[0], wbounds[1], n_pts_model)

    # Load selected extended background spectrum (units ph/s/m2/as2/um) for wavelength range which overfills mosaic
    # f_units_ext_in = 'phot/s/m2/um/arcsec2'
    w_ext, f_ext_in, f_units_ext_in = None, None, None
    tau_metis = 0.1 * Detector.qe
    if ext['sed'] == 'bb':
        tau_bb = 0.1 * 1.0  # Assume 10 % integrating sphere and an attenuation setting
        tbb = ext['temperature']
        w_ext, f_bb, f_units_ext_in = build_bb_emission(wave_range=wbounds, tbb=tbb)
        f_ext_in = tau_metis * tau_bb * f_bb
    if ext['sed'] == 'sky':
        w_ext, f_sky, f_units_ext_in = load_sky_emission(wave_range=wbounds)
        f_ext_in = tau_metis * f_sky
    if ext['sed'] == 'laser':
        w_ext, f_laser, f_units_ext_in = build_laser_emission(ext, wave_range=wbounds)
        f_ext_in = tau_metis * f_laser

    atel = math.pi * (39. / 2)**2  # ELT collecting area
    alpha_pix = Globals.alpha_mas_pix / 1000.  # Along slice pixel scale
    beta_slice = Globals.beta_mas_pix / 1000.  # Slice width
    delta_w = wave_cen / 100000  # Spectral resolution
    pix_delta_w = 2.5  # Pixels per spectral resolution element
    f_ext = f_ext_in * Detector.qe * atel * alpha_pix * beta_slice * delta_w / pix_delta_w  # el/pixel/second
    print('- converted to el/sec/pix')

    w_pnh, f_pnh_in, f_pnh_units = build_bb_emission(wave_range=wbounds, tbb=1000.)
    tau_wcu_attenuator = 1E-4
    f_pnh = f_pnh_in * tau_metis * tau_wcu_attenuator * atel * alpha_pix * beta_slice * delta_w / pix_delta_w  # el/pixel/second

    affines = filer.read_fits_affine_transform(date_stamp)
    svd_transforms = filer.read_fits_svd_transforms()

    n_det_rows_slice = 200          # Number of rows per slice, with comfortable margin...

    image_mosaic, waves_mosaic = [], []
    for det_no in range(1, 5):
        image_mosaic.append(np.zeros(det_shape))
        waves_mosaic.append(np.zeros(det_shape))

    opt_transforms, ech_orders = Util.find_optimum_transforms(wave_cen, opticon, svd_transforms)

    # Set up dictionary of blaze wavelengths from ech_angle=0 transforms
    blaze = make_blaze_dictionary(opt_transforms, opticon)

    for eo_slice_id in list(opt_transforms):
        txt = "cfg-ech_ord-slice_no-spifu_no= {:s}, ".format(eo_slice_id)

        opt_transform = opt_transforms[eo_slice_id]
        cfg = opt_transform['configuration']
        ech_ord = cfg['ech_ord']
        ech_ang = cfg['ech_ang']
        pri_ang = cfg['pri_ang']
        slice_no = cfg['slice']
        txt += "pri_ang= {:5.3f}, ech_ang= {:5.3f}, ".format(pri_ang, ech_ang)
        w_blaze, tau_blaze = make_tau_blaze(blaze, ech_ord, ech_ang)  # Make echelle blaze profile for this order.
        idx_max = np.argmax(tau_blaze)
        tau_blaze_max = tau_blaze[idx_max]
        w_blaze_max = w_blaze[idx_max]
        txt += "blaze wavelength= {:5.3f}".format(w_blaze_max)
        # Load PSF library.  PSFs are sampled at 4x detector resolution so need to be down-sampled.
        # To start with, we just use PSFs for the boresight slice (slice no. 13)
        t_now = time.perf_counter()
        t_min = (t_now - t_start) / 60.
        psf_dict = load_psf_dict(opticon, ech_ord, downsample=True)
        _, psf_ext = psf_dict[0]
        w_min, w_max = cfg['w_min'], cfg['w_max']
        yc = Util.slice_to_efp_y(slice_no, 0.)  # Slice y (beta) coordinate in EFP
        efp_y = np.array([yc, yc])
        efp_x = np.array([0., 0.])              # Slice x (alpha) bounds in EFP, map to dfp_y
        efp_w = np.array([w_min, w_max])
        efp_slice = {'efp_y': efp_y, 'efp_x': efp_x, 'efp_w': efp_w}    # Centre of slice for detector
        if '6_23_12_2' in eo_slice_id:
            nob = 1
        mfp_slice, oob = Util.efp_to_mfp(opt_transform, efp_slice)
        # if oob:
        #     print(txt + ' - all of spectrum is out of bounds at mosaic')
        #     continue
        dfp_slice = Util.mfp_to_dfp(affines, mfp_slice)
        det_row_min = int(dfp_slice['dfp_y'][0] - 100)     # Bracket slices which typically cover 120 rows.
        det_row_max = det_row_min + n_det_rows_slice
        txt += " - adding extended illumination,"
        print(txt)
        strip_shape = n_det_rows_slice, n_det_cols

        for det_no in dfp_slice['det_nos']:
            det_idx = det_no - 1
            ext_sig = np.zeros(strip_shape)
            psf_sig = np.zeros(strip_shape)

            dfp_det_nos = np.full(n_det_cols, det_no)
            dfp_pix_xs = np.arange(n_det_cols)  # Detector column indices

            n_rows_written = 0
            w_illuminated = []
            for det_row in range(det_row_min, det_row_max + 1):
                strip_row = det_row - det_row_min
                dfp_pix_ys = np.full(n_det_cols, det_row)
                dfp_row = {'dfp_x': dfp_pix_xs, 'dfp_y': dfp_pix_ys, 'det_nos': dfp_det_nos}
                efp_row = Util.dfp_to_efp(opt_transform, affines, dfp_row)
                efp_x = efp_row['efp_x']
                idx_illum = np.argwhere(np.abs(efp_x) < xh)
                n_ib = len(idx_illum)
                if n_ib == 0:  # Skip unilluminated rows.
                    continue
                # Apply extended spectrum (sky or black body) to illuminated rows
                efp_w_row = efp_row['efp_w']
                w_obs = efp_w_row[idx_illum][:]
                tau_echelle = np.interp(w_obs, w_blaze, tau_blaze)
                f_ext_obs = np.interp(w_obs, w_ext, f_ext)
                ext_sig[strip_row, idx_illum] = f_ext_obs * tau_echelle
                waves_mosaic[det_idx][det_row, idx_illum] = w_obs
                w_illuminated = w_illuminated + list(w_obs[:])
                n_rows_written += 1
            w_ill_np = np.array(w_illuminated)
            w_ext_min, w_ext_max = np.nanmin(w_ill_np), np.nanmax(w_ill_np)

            fmt = " - for detector {:d}, rows= {:d}-{:d}, w_min, w_max= {:4.3f} {:4.3f} at t= {:5.2f} min"
            txt = fmt.format(det_no, det_row_min, det_row_max, w_ext_min, w_ext_max, t_min)
            print(txt)

            psf_pnh = None
            if pinhole is not None:                             # Get dictionary of pinhole PSFs
                slice_no_pnh_cen = pinhole['slice_no_cen']
                slice_no_offset = slice_no - slice_no_pnh_cen
                sno_radius = 5 if opticon == nominal else 2
                if math.fabs(slice_no_offset) < sno_radius:     # Pinhole PSF exists for this slice.
                    _, psf_pnh = psf_dict[slice_no_offset]

            if psf_pnh is not None:                             # Add pinhole spectrum.
                print()
                print("Applying pinhole spectrum")

                efp_x_pnh = np.full(n_det_cols, pinhole['efp_x'])
                efp_y_pnh = np.full(n_det_cols, pinhole['efp_y'])
                efp_w_pnh = efp_w_row

                efp_pnh = {'efp_x': efp_x_pnh, 'efp_y': efp_y_pnh, 'efp_w': efp_w_pnh}
                dfp_pnh = Util.efp_to_dfp(opt_transform, affines, det_no, efp_pnh)
                # Row by row population of psf_illum image
                dfp_y_pnh = dfp_pnh['dfp_y']
                dfp_rows_pnh = np.round(dfp_y_pnh).astype(int)
                det_row_min_psf, det_row_max_psf = np.amin(dfp_rows_pnh), np.amax(dfp_rows_pnh)
                for det_row in range(det_row_min_psf, det_row_max_psf + 1):
                    dfp_pix_ys = np.full(n_det_cols, det_row)
                    idx_illum = np.argwhere(dfp_rows_pnh == det_row)
                    n_ib = len(idx_illum)
                    if n_ib == 0:               # Skip unilluminated rows.
                        continue
                    # Apply extended spectrum (sky or black body) to illuminated rows
                    strip_row = det_row - det_row_min
                    w_obs = efp_w_row[idx_illum][:]
                    f_pnh_obs = np.interp(w_obs, w_pnh, f_pnh)
                    tau_echelle = np.interp(w_obs, w_blaze, tau_blaze)
                    psf_sig[det_row, idx_illum] = f_pnh_obs * tau_echelle

            # Now convolve flux maps with psfs.
            image = image_mosaic[det_idx]
            image[det_row_min:det_row_max, :] += scipy.signal.convolve2d(ext_sig, psf_ext, mode='same', boundary='symm')
            if psf_pnh is not None:
                image[det_row_min:det_row_max, :] += scipy.signal.convolve2d(psf_sig, psf_pnh, mode='same', boundary='symm')
            # image_mosaic.append(image)

    # Add dark current and read noise(Finger, Rauscher)
    frame_mosaic = []
    for det_no in range(1, 5):
        det_idx = det_no - 1
        image = image_mosaic[det_idx]
        frame = Detector.detect(image, dit)
        frame_mosaic.append(frame)

    print()
    fits_out_path = '../output/' + obs_name + '.fits'
    print("Writing fits file - {:s}".format(fits_out_path))
    filer.write_fits(fits_out_path, hdr_primary, frame_mosaic)

    fits_out_path = '../output/' + obs_name + '_illumination.fits'
    print("Writing fits file - {:s}".format(fits_out_path))
    filer.write_fits(fits_out_path, hdr_primary, image_mosaic)

    fits_waves_out_path = '../output/' + obs_name + '_waves.fits'
    print("Writing fits file - {:s}".format(fits_waves_out_path))
    filer.write_fits(fits_waves_out_path, hdr_primary, waves_mosaic)
    print('lmssim done. ')
    return


def make_blaze_dictionary(transforms, opticon):
    blaze = {}
    for key in transforms:
        transform = transforms[key]
        cfg = transform['configuration']
        if cfg['slice'] != 13:
            continue
        # if opticon == Globals.spifu:
        #     if cfg['spifu'] != 1:
        #         continue
        ech_ang = cfg['ech_ang']
        mfp_bs = {'mfp_x': [0.], 'mfp_y': [0.]}
        ech_ord = cfg['ech_ord']
        efp_bs = Util.mfp_to_efp(transform, mfp_bs)
        wave = efp_bs['efp_w'][0]
        if ech_ord not in blaze:
            blaze[ech_ord] = {}
        blaze[ech_ord][ech_ang] = wave
    return blaze


def load_psf_dict(opticon, ech_ord, downsample=False, slice_no_tgt=13):
    analysis_type = 'iq'

    nominal = Globals.nominal
    nom_iq_date_stamp = '2024073000'
    nom_config = (analysis_type, nominal, nom_iq_date_stamp,
                  'Nominal spectral coverage (fov = 1.0 x 0.5 arcsec)',
                  None, None)

    spifu = Globals.spifu
    spifu_date_stamp = '2024061802'
    spifu_config = (analysis_type, spifu, spifu_date_stamp,
                    'Extended spectral coverage (fov = 1.0 x 0.054 arcsec)',
                    None, None)

    model_configurations = {nominal: nom_config, spifu: spifu_config}
    model_config = model_configurations[opticon]
    filer = Filer(model_config)
    defoc_str = '_defoc000um'

    _, _, date_stamp, _, _, _ = model_config
    dataset_folder = '../data/iq/' + opticon + '/' + date_stamp + '/'
    config_no = 41 - ech_ord if opticon == nominal else 0
    config_str = "_config{:03d}".format(config_no)

    psf_sum = 0.
    # Find a slice number
    nom_slice_no_rep_field = {1: (9, 17)}

    psf_dict = {}  # Create a new set of psfs

    # Use the boresight field position (field_no = 1) for now...
    (fn_min, fn_max) = (1, 2) if opticon == nominal else (1, 4)
    for field_no in range(fn_min, fn_max):
        field_idx = field_no - 1
        field_str = "_field{:03d}".format(field_no)
        iq_folder = 'lms_' + date_stamp + config_str + field_str + defoc_str
        spec_no = 0
        sn_radius = 4 if opticon == nominal else 1
        sn_min, sn_max = slice_no_tgt - sn_radius, slice_no_tgt + sn_radius + 1

        if opticon == spifu:
            # field_idx = field_no - 1
            spec_no = 1
            sn_min = slice_no_tgt - 1 + field_idx % 3
            sn_max = sn_min + 1

        for slice_no in range(sn_min, sn_max):
            iq_slice_str = "_spat{:02d}".format(slice_no) + "_spec{:d}_detdesi".format(spec_no)
            iq_filename = iq_folder + iq_slice_str + '.fits'
            iq_path = iq_folder + '/' + iq_filename
            file_path = dataset_folder + iq_path
            hdr, psf = filer.read_fits(file_path)
            # print("slice_no={:d}, psf_max={:10.3e}".format(slice_no, np.amax(psf)))
            if downsample:
                oversampling = 4
                n_psf_rows, n_psf_ncols = psf.shape
                n_ds_rows, n_ds_cols = int(n_psf_rows / oversampling), int(n_psf_ncols / oversampling)
                psf = psf.reshape(n_ds_rows, oversampling, n_ds_cols, -1).mean(axis=3).mean(axis=1)   # down sample
            slice_no_offset = slice_no - slice_no_tgt
            psf_dict[slice_no_offset] = hdr, psf
            psf_sum += np.sum(psf)
        # Normalise the PSFs to have unity total flux in detector space
        for slice_no in range(sn_min, sn_max):
            slice_no_offset = slice_no - slice_no_tgt
            _, psf = psf_dict[slice_no_offset]
            norm_factor = oversampling * oversampling / psf_sum
            psf *= norm_factor
    return psf_dict

@staticmethod
def build_laser_emission(laser, wave_range=[2.7, 5.7]):
    tau_bb = 0.1 * 1.0

    wmin, wmax = wave_range[0], wave_range[1]
    delta_w = wmin / 200000
    waves = np.arange(wmin, wmax, delta_w)
    flux = np.zeros(waves.shape)
    if 'nlines' in laser:
        line_waves = list(np.linspace(laser['wshort'], laser['wlong'], laser['nlines']))
    else:
        line_waves = [laser['wavelength']]
    f_laser = laser['flux']
    for line_wave in line_waves:
        woff = np.abs(waves - line_wave / 1000.)
        las_idx = np.argwhere(woff < delta_w)
        flux[las_idx] = f_laser
    units = 'phot/s/m2/um/arcsec2'
    return waves, flux, units


@staticmethod
def build_bb_emission(wave_range=[2.7, 5.7], tbb=1000.):
    hp = 6.626e-34
    cc = 2.997e+8
    kb = 1.38e-23
    as2_sterad = 4.25e10
    m_um = 1.e-6
    wmin, wmax = wave_range[0], wave_range[1]
    delta_w = wmin / 200000
    waves = np.arange(wmin, wmax, delta_w)
    flux = np.zeros(waves.shape)
    wm = waves * m_um
    a = hp * cc / (wm * kb * tbb)
    inz = np.argwhere(a < 400.)         # Set UV catastrophe points to zero
    b = np.exp(a[inz]) - 1.
    c = 2 * cc * np.power(wm[inz], -4)
    flux[inz] = m_um * (c / b) / as2_sterad

    units = 'phot/s/m2/um/arcsec2'
    return waves, flux, units


def make_tau_blaze(blaze, ech_ord, ech_ang):
    """ Generate a blaze profile (wavelength v efficiency) for an echelle order.
    I = sinc^2(pi (w - w_blaze) / w_width), where w_width = 0.042 w_blaze from SPIE model
    """
    w_n = blaze[ech_ord][ech_ang]
    w_wid = 0.042 * w_n
    n_pts = 500
    w_lo = w_n - 5. * w_wid
    w_hi = w_n + 5. * w_wid
    waves = np.linspace(w_lo, w_hi, n_pts)
    tk = 0.7 * np.power(np.sinc(math.pi * (w_n - waves) / w_wid), 2)
    return waves, tk

@staticmethod
def load_sky_emission(waves_in=None, wave_range=None):
    path = '../data/elt_sky.fits'
    hdu_list = fits.open(path, mode='readonly')
    data_table = hdu_list[1].data
    waves_all = data_table['lam'] / 1000.
    flux_all, flux_errs = data_table['flux'], None
    flux_units, label, colour = 'ph/s/m2/um/arcsec2', 'Emission', 'orange'
    print("Loaded sky transmission and emission spectrum with units {:s}".format(flux_units))
    if waves_in is not None:        # Interpolate spectrum onto input wavelength list
        flux_new = np.zeros(waves_in.shape)
        i = 0
        fmt = "{:10s}{:10s}{:10s}{:10s}{:10s}{:10s}"
        print(fmt.format('New Wave', 'New T', 'W1', 'W2', 'T1', 'T2'))
        for j, new_wave in enumerate(waves_in):
            while waves_all[i] < new_wave:
                i += 1
            flux_new[j] = np.interp(new_wave, waves_all[i - 1:i + 1], flux_all[i - 1:i + 1])
            i += 1
        return waves_in, flux_new, flux_units

    if wave_range is not None:
        wmin, wmax = wave_range[0], wave_range[1]
        idx1 = np.argwhere(wmin < waves_all)
        idx2 = np.argwhere(waves_all < wmax)
        idx = np.intersect1d(idx1, idx2)
        waves, flux = waves_all[idx], flux_all[idx]
        return waves, flux, flux_units

    flux_units = 'phot/s/m2/um/arcsec2'
    return waves_all, flux_all, flux_units

run()
