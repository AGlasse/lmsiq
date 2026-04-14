#!/usr/bin/env python
"""
Map observations taken (simulated or taken during AIV) against data files.  Used to set up simulations (in 'lmssim')
and also for analysis scripts (in 'lmsaiv').
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
        sim_configs, obs_keys = None, None
        for line in lines:
            if sim_configs is None:
                sim_configs = {}
                obs_keys = line.split(',')
                continue
            tokens = line.split(',')
            if '#' in tokens[0]:        # Skip commented lines.
                continue
            obs_cfg = {}
            for obs_key, token in zip(obs_keys, tokens):
                if obs_key == '':
                    continue
                obs_cfg[obs_key] = token
            cfg_id = obs_cfg['id']
            if test_name in cfg_id:
                in_csv_file = True
                sim_configs[cfg_id] = obs_cfg

        if not in_csv_file:
            print("!! Test {:s} not found in csv file {:s}".format(test_name, cfg_path))
            return None
        return sim_configs
