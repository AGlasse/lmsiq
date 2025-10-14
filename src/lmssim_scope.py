#!/usr/bin/env python
import sys
import numpy as np
import scopesim as sim
from scopesim import Source
from matplotlib import pyplot as plt
import synphot
from synphot import units as s_units
from astropy import units as u
import astropy.io.fits as fits
from lms_globals import Globals
from lmssim_model import Model


class Scope:

    def __init__(self):
        install_notebooks = False
        if install_notebooks:
            sim.download_packages(["METIS", "ELT", "Armazones"], release="latest")
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
            # cmds["!WCU.config_file"] = "./inst_pkgs/lms_ait_wcu_config.yaml"  # From metis_wcu_config

            pup_trans = 0. if sim_config['lms_pp1'] == 'closed' else 1.
            cmds["!OBS.pupil_transmission"] = pup_trans

            metis = sim.OpticalTrain(cmds)
#            splist = metis['lms_spectral_traces']
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
                result = metis.readout(dit=cmds['!OBS.dit'], ndit=cmds['!OBS.ndit'])[0]                 # detector_readout_mode="auto"
                obs_tag = "_{:03d}.fits".format(obs_idx)
                out_path = '../data/test_scopesim/' + obs_name + obs_tag
                result.writeto(out_path, overwrite=True)

        print('Done metsim')
        return

    # def setup(self, sim_cfg, debug=False):
    #     """ Set up observation command parameters.  Sources are defined by ImgHDS data cubes.
    #
    #     """
    #     cmds = sim.UserCommands(use_instrument="METIS", set_modes=["wcu_lms"])
    #     cmds['!OBS.dit'] = float(sim_cfg['dit'])             # Integration time per ramp (> 1.3 seconds)
    #     cmds['!OBS.ndit'] = int(sim_cfg['ndit'])             # No. of ramps
    #     wavelen = float(sim_cfg['wave_cen']) / 1000.
    #     cmds['!OBS.wavelen'] = wavelen                       # Observation wavelength (microns)
    #     src_config = Model.bgd_srcs[sim_cfg['bgd_src']]
    #     # if obs_cfg['lms_pp1'] == 'closed':
    #     #     src_config = Model.bgd_srcs['dark']
    #     bgd_source = None
    #     # bgd_source = self.create_source(src_config)       # All source information now in ScopeSim
    #     # Show all settable parameters (eg, cmd[param] = new_value
    #     if debug:
    #         print('Instrument modes')
    #         for mode in cmds.modes_dict:
    #             print(mode)
    #         fmt = "{:<40s}{:s}"
    #         print(fmt.format('param', 'value'))
    #         for param in cmds:
    #             print(fmt.format(param, str(cmds[param])))
    #     return cmds, bgd_source
    #
    # @staticmethod
    # def create_source(src_cfg, debug=False):
    #     # Create a 1 x 1 arcsec source (to slightly overfill the LMS EFP), sampled at half the along-slice plate scale.
    #     img_pitch = Globals.alpha_pix / 2.         # EFP image pitch in mas.
    #     img_fov = 1. * u.arcsec
    #     nxy = int(img_fov / img_pitch)
    #     img_shape = nxy, nxy
    #     cdelt_deg = img_pitch.to(u.deg)
    #     # Create a common fits header.
    #     hdr = fits.Header(dict(NAXIS=2, NAXIS1=nxy + 1, NAXIS2=nxy + 1,
    #                            CRPIX1=nxy / 2, CRPIX2=nxy / 2, CRVAL1=0,
    #                            CRVAL2=0, CDELT1=cdelt_deg.value, CDELT2=cdelt_deg.value,
    #                            CUNIT1="DEG", CUNIT2="DEG",
    #                            CTYPE1='RA---TAN', CTYPE2='DEC--TAN'))
    #     waves = np.arange(2700, 5500, 1) * u.nm     # Common wavelength range for source spectrum
    #
    #     img = None
    #     sed = src_cfg['sed']
    #     if sed == 'dark':
    #         img = np.full(img_shape, 1.)
    #         flux = np.zeros(waves.shape)
    #         sp = synphot.SourceSpectrum(synphot.Empirical1D, points=waves, lookup_table=flux)
    #     if sed == 'bb':
    #         img = np.full(img_shape, 1.)       # Pixel weights (uniform illumination).
    #         tbb = src_cfg['temperature']
    #         bb = Model.black_body(waves, tbb=tbb)    # ph/sec/Angstrom/cm2/sterad
    #         img_omega = img_fov.to(u.rad) * img_fov.to(u.rad)
    #         tau = src_cfg['tau']
    #         # a1 = bb * img_omega.value
    #         # a2 = tau * a1
    #         kludge = 0.0
    #         flux = kludge * tau * bb * img_omega.value     # Total flux in image, so normalise by n_pixels to get pixel signal.
    #         sp = synphot.SourceSpectrum(synphot.Empirical1D, points=waves, lookup_table=flux)
    #     if sed == 'laser':
    #         line_flux = src_cfg['flux']
    #         gwid = 50       # Gaussian width in pixels
    #         x, y = np.meshgrid(np.arange(nxy), np.arange(nxy))
    #         img = np.exp(-1 * (((x - nxy/2) / gwid) ** 2 + ((y - nxy/2) / gwid) ** 2))
    #     if sed == 'sky':
    #         img = np.full(img_shape, 1.)       # Pixel weights (uniform illumination).
    #         flux = Model.load_sky_emission(waves)
    #         sp = synphot.SourceSpectrum(synphot.Empirical1D, points=waves, lookup_table=flux)
    #         # print('Sky source spectrum not yet implemented !!')
    #
    #     # Create source as an ImageHDU object
    #     hdu = fits.ImageHDU(data=img, header=hdr)
    #     src = Source(image_hdu=hdu, spectra=sp)
    #     if debug:
    #         plt.imshow(src.fields[0].data)
    #         plt.show()
    #         src.spectra[0].plot()
    #         plt.show()
    #     return src

    # def convert_mjy_to_photlam(self, waves_um, flux_mjy):
    #     """ Convert from um, mJy to PHOTLAM flux units (ph s-1 cm-2 ang-1) and wavelengths in Angstroms.
    #     """
    #     m_um = 1.E-6
    #     waves_m = np.array(waves_um) * m_um
    #     flux_w_m2_hz = 1.E-29 * np.array(flux_mjy) * .000001
    #     cc = 2.997E+8
    #     flux_w_m2_m = flux_w_m2_hz * cc / (waves_m * waves_m)
    #     hc = 1.98645E-8
    #     ph_energy_j = hc / waves_m
    #     flux_ph_s_m2_m = ph_energy_j * flux_w_m2_m
    #     ang_m = 1.E10
    #     cm2_m2 = 1.E4
    #     flux_photlam = flux_ph_s_m2_m * ang_m * cm2_m2 * s_units.PHOTLAM
    #     waves_ang = waves_m * ang_m
    #     return waves_ang, flux_photlam
    #
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
