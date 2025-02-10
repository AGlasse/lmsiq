#!/usr/bin/env python
"""
Map observations taken (simulated or taken during AIV) against data files.  Used to setup simulations (in 'lmssim')
and also for analysis scripts (in 'lmsopt').
"""
from lms_globals import Globals


class ObsMap:

    def __init__(self):
        return

    @staticmethod
    def get_configuration(test_name):
        """ Read configuration dictionary for a specific test from /config/lms-opt-config.csv
        """
        cfg_path = '../config/lms-opt-config.csv'
        lines = open(cfg_path, 'r').read().splitlines()
        sim_config, obs_keys = None, None
        for line in lines:
            if sim_config is None:
                sim_config = {}
                obs_keys = line.split(',')
                continue
            tokens = line.split(',')
            obs_cfg = {}
            for obs_key, token in zip(obs_keys, tokens):
                if obs_key == '':
                    continue
                obs_cfg[obs_key] = token
            cfg_id = obs_cfg['id']

            if test_name in cfg_id:
                sim_config[cfg_id] = obs_cfg

            if test_name == 'lms_opt_01_t3':  # Create sim_config programmatically (its a biggy)
                print('Creating simulation configuration for ' + test_name)
                sim_config = ObsMap.get_opt_01_t3_config(test_name)
        return sim_config

    @staticmethod
    def get_opt_01_t3_config(test_name):
        sim_config = {}
        wave_setting = {'wcu_lt': [4680, 4690, 4700], 'wcu_ll': [5240], 'wcu_ls': [3390]}
        fmt1 = "{:s}_{:s}_{:s}_fpm_{:s}_{:4d}_{:04d}"
        for opticon in [Globals.nominal, Globals.spifu]:
            common_parameters = {'dit': '5.', 'ndit': '12', 'lms_pp1': 'open', 'opticon': opticon}
            for bgd_src in ['wcu_lt', 'wcu_ll', 'wcu_ls']:
                fp_mask = 'open'
                for wave in wave_setting[bgd_src]:
                    for beta in [0, -150, +150]:
                        efp_y = beta / 100.  # FIX THIS!!
                        obs_id = fmt1.format(test_name, opticon, bgd_src, fp_mask, wave, beta)
                        obs_cfg = common_parameters
                        obs_cfg['id'] = obs_id
                        obs_cfg['bgd_src'] = bgd_src
                        obs_cfg['efp_x'] = str(0.)
                        obs_cfg['efp_y'] = str(efp_y)
                        obs_cfg['fp_mask'] = fp_mask
                        obs_cfg['wave_cen'] = str(wave)
                        sim_config[obs_id] = obs_cfg
        return sim_config
