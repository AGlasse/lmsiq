#!/usr/bin/env python
import scopesim as sim


class Scope:

    def __init__(self):
        install_notebooks = False
        if install_notebooks:
            sim.download_packages(["METIS", "ELT", "Armazones"], release="latest")
        # path = os.path.abspath("E:/scopesim_inst_pkgs/inst_pkgs/")
        # sim.set_inst_pkgs_path(path)
        return

    def run(self, sim_configs):
        for obs_name in sim_configs:
            sim_config = sim_configs[obs_name]
            opticon = sim_config['opticon']
            mode = {'nominal': 'wcu_lms', 'extended': 'wcu_lms_extended'}[opticon]
            cmds = sim.UserCommands(use_instrument="METIS", set_modes=[mode])
            cmds['!OBS.dit'] = float(sim_config['dit'])  # Integration time per ramp (> 1.3 seconds)
            cmds['!OBS.ndit'] = int(sim_config['ndit'])  # No. of ramps
            wavelen = float(sim_config['wave_cen']) / 1000.
            cmds['!OBS.wavelen'] = wavelen  # Observation wavelength (microns)

            pup_trans = 0. if sim_config['lms_pp1'] == 'closed' else 1.
            cmds["!OBS.pupil_transmission"] = pup_trans

            metis = sim.OpticalTrain(cmds)
            wcu = metis['wcu_source']

            wcu_aper = float(sim_config['wcu_aper'])
            wcu_mask = sim_config['wcu_mask']
            wcu_angle = float(sim_config['wcu_angle'])
            wcu_x = float(sim_config['wcu_x'])
            wcu_y = float(sim_config['wcu_y'])
            wcu_shift = wcu_x, wcu_y

            wcu.set_bb_aperture(wcu_aper)
            wcu.set_fpmask(maskname=wcu_mask, angle=wcu_angle, shift=wcu_shift)
            print(wcu)

            metis.effects.pprint_all()
            n_obs = int(sim_config['nobs'])
            for obs_idx in range(0, n_obs):
                metis.observe()
                result = metis.readout(dit=cmds['!OBS.dit'], ndit=cmds['!OBS.ndit'])[0]
                obs_tag = "_o{:03d}.fits".format(obs_idx)
                out_path = '../data/test_scopesim/' + obs_name + obs_tag
                result.writeto(out_path, overwrite=True)

        print('Done metsim')
        return

    @staticmethod
    def make_exo_planet():
        # Read in toy exo-planet spectrum (PSG HR8799c, with CO and H2O atmosphere only. SRP = 2E5.)
        # Any 2 column spectrum will do...
        waves_um, flux_mjy = [], []
        data_file = '../data/spectra/hr8799c_co_h2o_only_psg_rad.txt'
        with open(data_file, 'r') as text_file:
            records = text_file.read().splitlines()
            for record in records[13:]:
                tokens = record.split('  ')
                waves_um.append(float(tokens[0]))
                flux_mjy.append(float(tokens[1]))
        return waves_um, flux_mjy
