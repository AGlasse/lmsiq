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
    def _get_chop_tag(cxy_str):
        cxy_int = int(1000. * float(cxy_str))
        cxy_sgn = 'n' if cxy_int < 0 else 'p'
        cxy_tag = "{:s}{:03d}".format(cxy_sgn, abs(cxy_int))
        return cxy_tag

    @staticmethod
    def get_configuration(test_name):
        """ Read configuration dictionary for a specific test from /config/lms-opt-config.csv
        """
        in_csv_file = False
        cfg_path = '../config/METIS_Performance_Test_Sequence_LMS.csv'
        lines = open(cfg_path, 'r').read().splitlines()
        sim_configs, obs_keys, key_fmts = None, None, None
        for line in lines:
            tokens = line.split(',')
            if '#' in tokens[0]:        # Skip commented lines.
                continue
            if sim_configs is None:     # First uncommented line holds the observation keys
                sim_configs = {}
                obs_keys = line.split(',')
                continue
            sim_config = {}
            for obs_key, token in zip(obs_keys, tokens):
                obs_key = obs_key.replace('-', '_').lower()
                if obs_key == '' or 'img_' in obs_key:
                    continue
                sim_config[obs_key] = token.lower()
            opticon = sim_config['lms_msa']
            step_no = sim_config['step_no']

            cfg_tag = ''
            is_wcu_pnh = 'lm_pinhole' in sim_config['wcu_fp2_1'] or 'lm_grid' in sim_config['wcu_fp2_1']
            is_cfo_pnh = 'pnh-1' in sim_config['cfo_fp2']
            is_pnh = is_wcu_pnh or is_cfo_pnh
            is_wcu_bb = 'closed' not in sim_config['wcu_bb_ap_mask']
            if is_pnh and is_wcu_bb:
                cx_off_tag = ObsMap._get_chop_tag(sim_config['cfo_chop_off_x'])
                cy_off_tag = ObsMap._get_chop_tag(sim_config['cfo_chop_off_y'])
                cfg_tag += "iso_alpha_a{:s}_b{:s}".format(cx_off_tag, cy_off_tag)
            if sim_config['dpr_type'] == 'flat_lamp':
                cfg_tag += 'flat_'

            if sim_config['wcu_laser_tune'] == 'true':
                lt_off_str = sim_config['wcu_laser_tune_woff']
                lt_off_nm = int(1000. * float(lt_off_str))
                lt_off_sgn = 'm' if lt_off_nm < 0 else 'p'
                lt_off_tag = "{:s}{:03d}".format(lt_off_sgn, abs(lt_off_nm))
                cfg_tag += "iso_lambda_woff_{:s}nm_".format(lt_off_tag)

            cfg_id = test_name + '_' + step_no + '_' + opticon[0:3] + '_' + cfg_tag[0:-1]
            sim_config['cfg_id'] = cfg_id
            if test_name in sim_config['test_id']:
                in_csv_file = True
                sim_configs[cfg_id] = sim_config

        if not in_csv_file:
            print("!! Test {:s} not found in csv file {:s}".format(test_name, cfg_path))
            return None
        return sim_configs
