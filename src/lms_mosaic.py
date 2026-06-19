#!/usr/bin/env python
"""
@author: Alistair Glasse

Update:
"""
import copy

from astropy.io.fits import ImageHDU


class Mosaic:

    name, primary_hdr, hdu_list = None, None, None

    def __init__(self, **kwargs):
        return

    @staticmethod
    def copy_mosaic(mosaic, clear_data=False, copy_name=''):
        file_name, primary_hdr, hdus = mosaic
        moscopy_hdus = []
        for hdu in hdus:
            moscopy_hdr = copy.deepcopy(hdu.header)
            moscopy_data = copy.deepcopy(hdu.data)
            moscopy_hdu = ImageHDU(data=moscopy_data, header=moscopy_hdr)
            if clear_data:
                moscopy_hdu.data *= 0.
            moscopy_hdus.append(moscopy_hdu)
        moscopy_name = file_name if copy_name == '' else copy_name
        moscopy = moscopy_name, primary_hdr, moscopy_hdus
        return moscopy

    @staticmethod
    def diff_mosaics(mos1, mos2):
        mos_diff = Mosaic.copy_mosaic(mos1, clear_data=True, copy_name='diff_mosaic')
        _, _, hdus1 = mos1
        _, _, hdus2 = mos2
        _, _, hdus_diff = mos_diff
        for hdu1, hdu2, hdu in zip(hdus1, hdus2, hdus_diff):
            hdu.data = hdu1.data - hdu2.data
        return mos_diff

    @staticmethod
    def sum_mosaics(mos1, mos2):
        mos_sum = Mosaic.copy_mosaic(mos1, clear_data=True, copy_name='diff_mosaic')
        _, _, hdus1 = mos1
        _, _, hdus2 = mos2
        _, _, hdus_diff = mos_sum
        for hdu1, hdu2, hdu in zip(hdus1, hdus2, hdus_diff):
            hdu.data = hdu1.data + hdu2.data
        return mos_sum
