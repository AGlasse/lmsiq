#!/usr/bin/env python
import scopesim as sim
import astropy.units as u
from lms_filer import Filer


class Scope:

    def __init__(self):
        install_notebooks = False
        if install_notebooks:
            sim.download_packages(["METIS", "ELT", "Armazones"], release="latest")
        Filer.set_test_data_folder('scopesim')
        # path = os.path.abspath("E:/scopesim_inst_pkgs/inst_pkgs/")
        # sim.set_inst_pkgs_path(path)
        return

    def run(self, sim_configs):
        for obs_name in sim_configs:
            sim_config = sim_configs[obs_name]
            # print(sim_config)
            opticon = sim_config['opticon']
            mode = {'nominal': 'wcu_lms', 'extended': 'wcu_lms_extended'}[opticon]

            cmds = sim.UserCommands(use_instrument="METIS", set_modes=[mode])
            cmds['!OBS.dit'] = float(sim_config['dit'])     # Integration time per ramp (> 1.3 seconds)
            cmds['!OBS.ndit'] = int(sim_config['ndit'])     # No. of ramps
            wavelen = float(sim_config['wave_cen']) / 1000.
            cmds['!OBS.wavelen'] = wavelen                  # Observation wavelength (microns)
            pup_trans = 0. if sim_config['lms_pp1'] == 'closed' else 1.
            cmds["!OBS.pupil_transmission"] = pup_trans

            metis = sim.OpticalTrain(cmds)
            splist = metis['lms_spectral_traces']

            wcu = metis['wcu_source']
            wcu_aper = float(sim_config['wcu_aper'])
            wcu_mask = sim_config['wcu_mask']
            wcu_angle = float(sim_config['wcu_angle'])
            wcu_x = float(sim_config['wcu_x'])
            wcu_y = float(sim_config['wcu_y'])
            wcu_shift = wcu_x, wcu_y
            if sim_config['bgd_src'] == 'wcu_lt':
                metis["wcu_source"].set_lamp("laser")
            if sim_config['bgd_src'] == 'wcu_bb':
                metis["wcu_source"].set_lamp("bb")
                metis["wcu_source"].set_temperature(bb_temp=1000 * u.K, is_temp=320 * u.K, wcu_temp=295 * u.K)
            wcu.set_bb_aperture(value=wcu_aper)
            wcu.set_fpmask(maskname=wcu_mask, angle=wcu_angle, shift=wcu_shift)
            print(metis)
            metis.effects.pprint_all()
            print(wcu)
            n_obs = int(sim_config['nobs'])
            for obs_idx in range(0, n_obs):
                metis.observe()
                result = metis.readout(dit=cmds['!OBS.dit'], ndit=cmds['!OBS.ndit'])[0]
                obs_tag = "_o{:03d}.fits".format(obs_idx)
                # Add test configuration information explicitly to primary header.
                pri_hdr = result[0].header
                pri_hdr['ACHG WCU X'] = wcu_x
                pri_hdr['ACHG WCU Y'] = wcu_y
                pri_hdr['ACHG LASER WAVE'] = wavelen
                out_folder = '../data/test_scopesim'
                out_file = obs_name + obs_tag
                out_path = out_folder + '/' + out_file
                result.writeto(out_path, overwrite=True, checksum=True)

        print('Done simulating')
        return
