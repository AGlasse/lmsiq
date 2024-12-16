#!/usr/bin/env python
"""
Map observations taken (simulated or taken during AIV) against data files.  Used to setup simulations (in 'lmssim')
and also for analysis scripts (in 'lmsopt').
"""


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
        return sim_config
