#!/usr/bin/env python
"""

"""
import numpy as np
from lmsdist_util import Util
from lms_globals import Globals
from lms_filer import Filer
from lmssim_engine import Engine

tau_blaze_kernel = None     # Kernel blaze profile tau(x) where x = (wave / blaze_wave(eo) - 1)


def make_blaze_dictionary(transforms, opticon):
    blaze = {}
    for key in transforms:
        transform = transforms[key]
        cfg = transform['configuration']
        if cfg['slice'] != 13:
            continue
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
    dataset_folder = '../data/model/iq/' + opticon + '/' + date_stamp + '/'
    config_no = 41 - ech_ord if opticon == nominal else 0
    config_str = "_config{:03d}".format(config_no)

    psf_sum = 0.
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

engine = Engine()
engine.run()
