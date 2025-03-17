#!/usr/bin/env python
import math
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
        return

    def run(self, sim_cfg):
        for obs_name in sim_cfg:
            obs_cfg = sim_cfg[obs_name]
            if obs_cfg['opticon'] == Globals.extended:
                print('Skipping simulation ' + obs_name + ', extended spectral coverage not yet supported !!')
                continue
            cmd, source = self.setup(obs_cfg)
            metis = sim.OpticalTrain(cmd)
            metis['skycalc_atmosphere'].include = False         # Turn off sky and telescope for AIV tests
            metis['telescope_reflection'].include = False
            metis['cold_stop'].include = False                  # 67 K emission
            n_obs = int(obs_cfg['nobs'])                                     # = 5 for 5 fits files.
            for obs_idx in range(0, n_obs):
                metis.observe(source)
                result = metis.readout(detector_readout_mode="auto")[0]
                obs_tag = "_{:03d}.fits".format(obs_idx)
                out_path = '../data/test_scopesim/' + obs_name + obs_tag
                result.writeto(out_path, overwrite=True)

        print('Done metsim')
        return

    def setup(self, obs_cfg):
        """ Set up observation command parameters.  Sources are defined by ImgHDS data cubes.

        """
        model = Model()
        cmd = sim.UserCommands(use_instrument="METIS", set_modes=["lms"])
        cmd['!OBS.dit'] = float(obs_cfg['dit'])             # Integration time per ramp (> 1.3 seconds)
        cmd['!OBS.ndit'] = int(obs_cfg['ndit'])             # No. of ramps
        wavelen = float(obs_cfg['wave_cen']) *u.nm
        cmd['!OBS.wavelen'] = wavelen                       # Observation wavelength (microns)
        bgd_src = obs_cfg['bgd_src']
        wrange = 2.  # Width of wavelength range to model (4 um to overfill mosaic in spifu mode)
        wbounds = [wavelen - wrange / 2., wavelen + wrange / 2.]
        w_ext, f_ext = model.get_flux(Globals.scopesim, wbounds, bgd_src)
        waves_ang, flux_photlam = self.convert_mjy_to_photlam(w_ext, f_ext)
        src_config = Model.bgd_srcs[obs_cfg['bgd_src']]
        if obs_cfg['lms_pp1'] == 'closed':
            src_config = Model.bgd_srcs['dark']
        bgd_source = self.create_source(src_config)

        # efp_x, efp_y = obs_cfg['efp_x'],obs_cfg['efp_y']



        return cmd, bgd_source

    @staticmethod
    def create_source(src_cfg, debug=False):
        # Create a 1 x 1 arcsec source (to slightly overfill the LMS EFP), sampled at half the along-slice plate scale.
        img_pitch = Globals.alpha_pix / 2.         # EFP image pitch in mas.
        img_fov = 1.*u.arcsec
        nxy = int(img_fov / img_pitch)
        img_shape = nxy, nxy
        cdelt_deg = img_pitch.to(u.deg)
        # Create a common fits header.
        hdr = fits.Header(dict(NAXIS=2, NAXIS1=nxy + 1, NAXIS2=nxy + 1,
                               CRPIX1=nxy / 2, CRPIX2=nxy / 2, CRVAL1=0,
                               CRVAL2=0, CDELT1=cdelt_deg.value, CDELT2=cdelt_deg.value,
                               CUNIT1="DEG", CUNIT2="DEG",
                               CTYPE1='RA---TAN', CTYPE2='DEC--TAN'))
        wave = np.arange(27000, 55000, 10)      # Common wavelength range for source spectrum

        if src_cfg['sed'] == 'dark':
            img = np.full(img_shape, 1.)
            flux = np.zeros(wave.shape)
            sp = synphot.SourceSpectrum(synphot.Empirical1D, points=wave, lookup_table=flux)

        if src_cfg['sed'] == 'bb':
            img = np.full(img_shape, 1.)       # Pixel weights (uniform illumination).
            tbb = src_cfg['temperature']
            bb = Model.black_body(wave, tbb=tbb)    # ph/sec/Angstrom/cm2/sterad
            img_omega = img_fov.to(u.rad) * img_fov.to(u.rad)
            tau = src_cfg['tau']
            flux = tau * bb(wave) * img_omega.value     # Total flux in image, so normalise by n_pixels to get pixel signal.
            sp = synphot.SourceSpectrum(synphot.Empirical1D, points=wave, lookup_table=flux)

        if src_cfg['sed'] == 'laser':
            line_flux = src_cfg['flux']
            gwid = 50       # Gaussian width in pixels
            x, y = np.meshgrid(np.arange(nxy), np.arange(nxy))
            img = np.exp(-1 * (((x - nxy/2) / gwid) ** 2 + ((y - nxy/2) / gwid) ** 2))

        # Create ImageHDU object
        hdu = fits.ImageHDU(data=img, header=hdr)

        # Source creation
        src = Source(image_hdu=hdu, spectra=sp)

        if debug:
            plt.imshow(src.fields[0].data)
            plt.show()
            src.spectra[0].plot()
            plt.show()
        return src

    # @staticmethod
    # def import_source(sim, sources, spatial_extent, do_plot=True):
    #     """ Import spectrum into ScopeSim 'Source' object.
    #     """
    #     if spatial_extent == 'point':
    #         waves_ang, flux_photlam = sources[0]
    #         spec = SourceSpectrum(Empirical1D, points=waves_ang, lookup_table=flux_photlam)
    #         spectra = [spec]
    #         source = sim.Source(spectra=spectra, ref=np.zeros(1), x=np.array([0.]), y=np.array([0.]))
    #     if spatial_extent == 'extended':
    #         source = sources[0]
    #
    #     if do_plot:
    #         source.plot()
    #         plt.show()
    #         spectrum = source.spectra[0]
    #         spectrum.plot()
    #         plt.show()
    #     return source

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
