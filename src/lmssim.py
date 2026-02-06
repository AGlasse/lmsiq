#!/usr/bin/env python
"""

"""
from lms_globals import Globals
from lms_obs_map import ObsMap
from lmssim_scope import Scope
from lmssim_toy import Toy
import scopesim as sim

redownload_metis = False
sim_test_name = 'lms_opt_02'       # 'lms_opt_01_t2'
sim_id = Globals.scopesim_id
print('Simulating measurements for test {:s} using {:s}'.format(sim_test_name, sim_id))


def run(test_name):
    """ Run simulation, using either ScopeSim or 'ToySim' for Alistair's bespoke simulator
    - using LMS entrance focal plane to detector focal plane transforms, calculated in lmsdist.py
    - using LMS PSFs, currently including the (significant) optical design aberrations, which will be updated using
      real wavefront error maps.
    - implementing calibration sources (WCU and sea-level sky) available during AIT in Leiden.
    Start by defining the observation.
    """
    _ = Globals()
    scope = Scope()
    toy = Toy()
    obs_map = ObsMap()
    if redownload_metis:
        sim.download_packages(["METIS", "ELT", "Armazones"], release="latest")

    sim_config = obs_map.get_configuration(test_name)
    if sim_id == Globals.scopesim_id:
        scope.run(sim_config)
    if sim_id == Globals.toysim_id:
        toy.run(sim_config)
    return

run(sim_test_name)
