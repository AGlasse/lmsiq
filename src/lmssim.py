#!/usr/bin/env python
"""

"""
from lms_globals import Globals
from lms_obs_map import ObsMap
from lmssim_scope import Scope
from lmssim_toy import Toy
import scopesim as sim


def run():
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

    redownload_metis = False
    if redownload_metis:
        sim.download_packages(["METIS", "ELT", "Armazones"], release="latest")
    # ============================== Set simulator parameters here ====================================
    test_name = 'lms_opt_01'        # Name of test to simulate, e.g 'lms_opt_01'
    use_scope_sim = False

    base_debug_level = 'low'
    Globals.set_debug_level(base_debug_level)

    Globals.sim_id = Globals.scopesim_id if use_scope_sim else Globals.toysim_id
    print('Simulating measurements for test {:s} using {:s}'.format(test_name, Globals.sim_id))

    sim_config = obs_map.get_configuration(test_name)
    if sim_config is None:
        return
    if use_scope_sim:
        scope.run(sim_config)
    else:
        toy.run(sim_config)
    return

run()
