#!/usr/bin/env python
"""
Map observations taken (simulated or taken during AIV) against data files.  Used to set up simulations (in 'lmssim')
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
        in_csv_file = False
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
                in_csv_file = True
                if test_name == 'lms_opt_01_t3':  # Create sim_config programmatically (its a biggy)
                    print('Creating simulation configuration for ' + test_name)
                    simulator = obs_cfg['simulator']
                    sim_config = ObsMap.get_opt_01_t3_config(test_name, simulator)

                    return sim_config
                sim_config[cfg_id] = obs_cfg

        if not in_csv_file:
            print("!! Test {:s} not found in csv file".format(test_name))
            return None
        return sim_config

    @staticmethod
    def get_opt_01_t3_config(test_name, simulator):
        sim_config = {}
        obs_cfg = None
        wave_setting = {'wcu_lt': [4680 + 10*i for i in range(0, 8)],
                        'wcu_ll': [5240], 'wcu_ls': [3390]}
        fmt1 = "{:s}_{:s}_{:s}_fpm_{:s}_{:4d}_{:04d}"
        for opticon in [Globals.nominal, Globals.extended]:
            common_parameters = {'dit': '5.', 'ndit': '12', 'lms_pp1': 'open',
                                 'simulator': simulator, 'opticon': opticon, 'nobs': 1}
            for bgd_src in wave_setting:
                fp_mask = 'open'
                for beta in [0, -150, +150]:
                    efp_y = beta / 100.  # FIX THIS!!
                    waves = wave_setting[bgd_src]
                    for wave in waves:
                        obs_id = fmt1.format(test_name, opticon, bgd_src, fp_mask, wave, beta)
                        obs_parameters = {'id': obs_id, 'bgd_src': bgd_src,
                                          'efp_x': str(0.), 'efp_y': str(efp_y),
                                          'fp_mask': fp_mask, 'wave_cen': str(wave),
                                          }
                        obs_cfg = common_parameters | obs_parameters
                        # sim_config[obs_id] = obs_cfg
                    for alpha in [0, -300, 300]:
                        efp_x = alpha / 100.
                        for wave in waves:
                            obs_id = fmt1.format(test_name, opticon, bgd_src, fp_mask, wave, beta)
                            obs_parameters = {'id': obs_id, 'bgd_src': bgd_src,
                                              'efp_x': str(efp_x), 'efp_y': str(efp_y),
                                              'fp_mask': fp_mask, 'wave_cen': str(wave),
                                              }
                            obs_cfg = common_parameters | obs_parameters
                            # sim_config[obs_id] = obs_cfg
                    sim_config[obs_id] = obs_cfg
        return sim_config
