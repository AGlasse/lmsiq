#!/usr/bin/env python
"""

"""
from lms_globals import Globals
from lms_obs_map import ObsMap
from lmssim_scope import Scope
from lmssim_toy import Toy

test_name = 'lms_opt_01_t2'
force_simulator = Globals.scopesim   # Set = None to use simulator specified in lms-opt-config.csv


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

    sim_config = obs_map.get_configuration(test_name)
    for obs_name in sim_config:
        obs_cfg = sim_config[obs_name]
        simulator = obs_cfg['simulator']
        if force_simulator is not None:
            simulator = force_simulator
        if simulator == Globals.scopesim:
            scope.run(sim_config)
        if simulator == Globals.toysim:
            toy.run(sim_config)
    return

run(test_name)
