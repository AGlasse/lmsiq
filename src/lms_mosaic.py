#!/usr/bin/env python
"""
@author: Alistair Glasse

Update:
"""
import copy
from astropy.io.fits import ImageHDU


class Mosaic:

    name, primary_hdr, hdu_list = None, None, None

    def __init__(self, name, primary_hdr, hdu_list):
        self.name = name
        self.primary_hdr = primary_hdr
        self.hdu_list = hdu_list
        return

    def copy(self, clear_data=False, copy_name=''):
        mos_hdus = []
        for hdu in self.hdu_list:
            mos_hdr = copy.deepcopy(hdu.header)
            mos_data = copy.deepcopy(hdu.data)
            if clear_data:
                mos_data *= 0.
            mos_hdu = ImageHDU(data=mos_data, header=mos_hdr)
            mos_hdus.append(mos_hdu)
        mos_name = self.name if copy_name == '' else copy_name
        mos_pri_hdr = copy.deepcopy(self.primary_hdr)
        return Mosaic(mos_name, mos_pri_hdr, mos_hdus)

    def subtract(self, mos):
        for self_hdu, hdu in zip(self.hdu_list, mos.hdu_list):
            self_hdu.data = self_hdu.data - hdu.data
        return self

    def add(self, mos):
        for self_hdu, hdu in zip(self.hdu_list, mos.hdu_list):
            self_hdu.data = self_hdu.data + hdu.data
        return self
