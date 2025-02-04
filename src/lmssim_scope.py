#!/usr/bin/env python
import numpy as np
import scopesim as sim
from matplotlib import pyplot as plt
from synphot import SourceSpectrum, Empirical1D
from synphot import units as s_units
from astropy import units as u
from lms_globals import Globals
from lmssim_model import Model


class Scope:

    # sim.bug_report()      Crashes here.  Bug #491 submitted.
    def __init__(self):
        install_notebooks = False
        if install_notebooks:
            sim.download_packages(["METIS", "ELT", "Armazones"])

        # List all settable modes.
        cmd = sim.UserCommands(use_instrument="METIS", set_modes=["lms"])
        print('Instrument modes')
        for mode in cmd.modes_dict:
            print(mode)

    def run(self, obs_cfg):
        for obs_name in obs_cfg:
            cfg = obs_cfg[obs_name]
            cmd, sources = self.setup(cfg)
            metis = sim.OpticalTrain(cmd)
            field = self.import_source(sim, sources, 'extended')

            n_obs = 5  # For 5 minutes of data in 5 fits files.
            for obs_idx in range(0, n_obs):
                metis.observe(field)
                result = metis.readout(detector_readout_mode="auto")[0]
                out_name = "lms_test_obs_{:03d}.fits".format(obs_idx)
                result.writeto(out_name, overwrite=True)
        print('Done metsim')
        return

    def setup(self, obs_cfg):
        """ Set up observation command parameters.

        """
        if obs_cfg['opticon'] == Globals.spifu:
            print('Extended spectral coverage not yet supported !!')
            return
        model = Model()
        cmd = sim.UserCommands(use_instrument="METIS", set_modes=["lms"])
        cmd['!OBS.dit'] = float(obs_cfg['dit'])        # Integration time per ramp (> 1.3 seconds)
        cmd['!OBS.ndit'] = int(obs_cfg['ndit'])               # No. of ramps
        wavelen = float(obs_cfg['wave_cen']) / 1000.
        cmd['!OBS.wavelen'] = wavelen        # Observation wavelength (microns)
        cmd['!ATMO.pressure'] = 0.
        bgd_src = obs_cfg['bgd_src']
        wrange = 2.  # Width of wavelength range to model (4 um to overfill mosaic in spifu mode)
        wbounds = [wavelen - wrange / 2., wavelen + wrange / 2.]
        w_ext, f_ext, f_units_ext_in = model.get_flux(wbounds, bgd_src)
        waves_ang, flux_photlam = self.convert_mjy_to_photlam(w_ext, f_ext)
        spec = SourceSpectrum(Empirical1D, points=waves_ang, lookup_table=flux_photlam)
        source = w_ext, 100. * f_ext
        efp_x, efp_y = obs_cfg['efp_x'],obs_cfg['efp_y']
        xs, ys, pixel_scale = [-1., 1.], [-1, 1.], .01      # arcsec
        flux = 100 * u.Jy
        source = sim.source.source_templates.uniform_illumination(xs, ys, pixel_scale,
                                                                  flux=flux, spectrum=spec)
        return cmd, [source]

    @staticmethod
    def import_source(sim, sources, spatial_extent, do_plot=True):
        """ Import spectrum into ScopeSim 'Source' object.
        """
        if spatial_extent == 'point':
            waves_ang, flux_photlam = sources[0]
            spec = SourceSpectrum(Empirical1D, points=waves_ang, lookup_table=flux_photlam)
            spectra = [spec]
            source = sim.Source(spectra=spectra, ref=np.zeros(1), x=np.array([0.]), y=np.array([0.]))
        if spatial_extent == 'extended':
            source = sources[0]

        if do_plot:
            source.plot()
            plt.show()
            spectrum = source.spectra[0]
            spectrum.plot()
            plt.show()
        return source

    def convert_mjy_to_photlam(self, waves_um, flux_mjy):
        """ Convert from um, mJy to PHOTLAM flux units (ph s-1 cm-2 ang-1) and wavelengths in Angstroms.
        """
        m_um = 1.E-6
        waves_m = np.array(waves_um) * m_um
        flux_w_m2_hz = 1.E-29 * np.array(flux_mjy) * .000001
        cc = 2.997E+8
        flux_w_m2_m = flux_w_m2_hz * cc / (waves_m * waves_m)
        hc = 1.98645E-8
        ph_energy_j = hc / waves_m
        flux_ph_s_m2_m = ph_energy_j * flux_w_m2_m
        ang_m = 1.E10
        cm2_m2 = 1.E4
        flux_photlam = flux_ph_s_m2_m * ang_m * cm2_m2 * s_units.PHOTLAM
        waves_ang = waves_m * ang_m
        return waves_ang, flux_photlam

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
