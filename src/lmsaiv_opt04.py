#!/usr/bin/env python
"""
Decorators for use in all LMS projects.  Currently just includes @debug

@author: Alistair Glasse

Update:
"""
from lms_globals import Globals
from lmsaiv_plot import Plot
from lms_filer import Filer


class Opt04:


    def __init__(self):
        return

    @staticmethod
    def ghost(title, as_built, **kwargs):
        test_name = 'lms_opt_03'
        opticon = Globals.nominal

        ref_step = 'Step1'
        tgt_step = 'Step2'

        inc_tags = [test_name, opticon[0:3]]         # Tokens to identify image files.
        filer = Filer()
        filer.set_configuration('distortion', opticon, is_ait_data=True)

        flag_text = 'Analysing'
        # Load offset/background images to subtract from ghost images
        ref_mosaic = Filer.read_mosaic_list(Filer.test_data_folder, inc_tags + ref_step)[0]
        tgt_mosaic = Filer.read_mosaic_list(Filer.test_data_folder, inc_tags + tgt_step)[0]
        sig_mosaic = tgt_mosaic.subtract(ref_mosaic)
        Plot.mosaic(sig_mosaic, title='tgt - ref mosaic', cmap='hot')
        return
