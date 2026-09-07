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
    def is_dark(sim_config):
        is_dark = 'dark' in sim_config['cfo_fp2']
        is_dark = True if 'dark' in sim_config['cfo_pp1'] else is_dark
        is_dark = True if 'closed' in sim_config['lms_pp1'] else is_dark
        return is_dark

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
            if test_name not in tokens[0].lower():      # Select requested test
                continue
            sim_config = {}
            for obs_key, token in zip(obs_keys, tokens):
                obs_key = obs_key.replace('-', '_').lower()
                if obs_key == '' or 'img_' in obs_key:
                    continue
                sim_config[obs_key] = token.lower()

            opticon = sim_config['lms_msa']
            step_no = sim_config['step_no']
            is_dark = ObsMap.is_dark(sim_config)
            is_cfo_pnh = 'pnh-1' in sim_config['cfo_fp2']
            is_wcu_pnh = 'lm_pinhole' in sim_config['wcu_fp2_1'] or 'lm_grid' in sim_config['wcu_fp2_1']
            is_pnh = is_wcu_pnh or is_cfo_pnh
            is_tunable_laser = sim_config['wcu_laser_tune'] == 'true'
            is_sw_laser = sim_config['wcu_laser_tune'] == 'true'
            is_lw_laser = sim_config['wcu_laser_tune'] == 'true'
            is_laser = is_tunable_laser or is_sw_laser or is_lw_laser
            is_bb_on = sim_config['wcu_bb_ap_mask'] != 'closed'
            is_sky = sim_config['wcu_per_arm'] != 'out'

            cfg_tag = ''
            if is_dark:
                cfg_tag += 'dark_'
            else:
                if is_sky:
                    cfg_tag += 'sky_'
                if is_pnh:
                    if is_bb_on:
                        cx_off_tag = ObsMap._get_chop_tag(sim_config['cfo_chop_off_x'])
                        cy_off_tag = ObsMap._get_chop_tag(sim_config['cfo_chop_off_y'])
                        cfg_tag += "bb_iso_alpha_a{:4s}_b{:4s}_".format(cx_off_tag, cy_off_tag)
                    if is_laser:
                        cfg_tag += 'las_psf_'
                else:
                    if is_bb_on:
                        cfg_tag += 'flat_bb_'
                    if is_laser:
                        cfg_tag += 'las_iso_lambda_'
                if is_tunable_laser:
                    lt_woff_str = sim_config['wcu_laser_tune_woff']
                    lt_woff_nm = int(1000. * float(lt_woff_str))
                    lt_woff_sgn = 'm' if lt_woff_nm < 0 else 'p'
                    lt_woff_tag = "{:s}{:03d}".format(lt_woff_sgn, abs(lt_woff_nm))
                    cfg_tag += "woff_{:s}nm_".format(lt_woff_tag)

            cfg_id = test_name + '_' + step_no + '_' + opticon[0:3] + '_' + cfg_tag[0:-1]
            sim_config['cfg_id'] = cfg_id
            sim_configs[cfg_id] = sim_config

        if not sim_configs:         # No configurations found
            print("!! Test {:s} not found in csv file {:s}".format(test_name, cfg_path))
            return None
        return sim_configs
