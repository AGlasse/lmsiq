#!/usr/bin/env python
"""

"""
from lms_globals import Globals
from lms_obs_map import ObsMap
from lmssim_scope import Scope
from lmssim_toy import Toy


def run(test_name, simulator='ScopeSim'):
    """ Run simulation, using either ScopeSim or 'ToySim' for Alistair's bespoke simulator
    - using LMS entrance focal plane to detector focal plane transforms, calculated in lmsdist.py
    - using LMS PSFs, currently including the (significant) optical design aberrations, which will be updated using
      real wavefront error maps.
    - implementing calibration sources (WCU and sea-level sky) available during AIT in Leiden.
    Start by defining the observation.
    """
    _ = Globals()
    obs_map = ObsMap()
    sim_config = obs_map.get_configuration(test_name)

    if simulator == 'ScopeSim':
        scope = Scope()
        scope.run(sim_config)

    if simulator == 'ToySim':
        toy = Toy()
        toy.run(sim_config)
    return


run('lms_opt_01_t3', simulator='ToySim')       # 'ScopeSim' or 'ToySim'
