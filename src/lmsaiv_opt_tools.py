import math
import numpy as np
import scipy
from scipy.optimize import curve_fit, OptimizeWarning
from lms_globals import Globals


class OptTools:


    def __init__(self):
        return

    @staticmethod
    def copy_mosaic(mosaic, clear_data=False, copy_name=''):
        file_name, hdr, hdus = mosaic
        moscopy_hdus = []
        for hdu in hdus:
            moscopy_hdu = hdu.copy()
            if clear_data:
                moscopy_hdu.data *= 0.
            moscopy_hdus.append(moscopy_hdu)
        moscopy_name = file_name if copy_name == '' else copy_name
        moscopy = moscopy_name, hdr, moscopy_hdus
        return moscopy

    @staticmethod
    def dark_stats(mosaics):
        for mosaic in mosaics:
            file_name, hdr, hdus = mosaic
            dit = hdr['HIERARCH ESO DET DIT']
            ndit = hdr['HIERARCH ESO DET NDIT']
            t_int = dit * ndit

            print()
            print("File = {:s}".format(file_name))
            fmt = "Signal distribution statistics, ndit= {:d} x dit= {:3.1f} sec, integration time = {:6.1f}"
            print(fmt.format(ndit, dit, t_int))
            fmt = "{:>8s},{:>10s},{:>10s},{:>10s},{:>10s}"
            print(fmt.format('Detector', 'median', 'stdev', 'median', 'Rd_Noise'))
            print(fmt.format('No.', 'DN', 'DN', 'el/sec.', 'el.'))
            fmt = "{:8d},{:10.3f},{:10.3f},{:10.3f},{:10.3f}"

            for i, hdu in enumerate(hdus):
                el_adu = float(hdu.header['HIERARCH ESO DET3 CHIP GAIN'])
                median = np.median(hdu.data)
                stdev = np.std(hdu.data)
                median_current = median * el_adu / dit
                rd_noise = stdev * el_adu / math.sqrt(2. / ndit)
                text = fmt.format(i + 1, median, stdev, median_current, rd_noise)
                print(text)
        return

    @staticmethod
    def transform_detector_image(mosaic, det_no, xy_pix=(0, 0), angle=0.0):
        mos_name, mos_primary_header, mos_hdus = mosaic
        det_idx = det_no - 1
        cosa = math.cos(math.radians(angle))
        sina = math.sin(math.radians(angle))
        img = mos_hdus[det_idx].data
        cval = np.median(img)
        tr_mat = np.array([[cosa, sina, 0.], [-sina, cosa, 0.], [0., 0., 1.]])
        rotimg = scipy.ndimage.affine_transform(img, tr_mat, cval=cval, order=3)
        mos_hdus[det_idx].data = rotimg
        return mosaic

    @staticmethod
    def _find_thetas(det_traces):
        """ Calculate the angle between each trace and the 'u' axis (row for iso-alpha traces, column for iso-lambda),
        evaluated from the gradient of the polynomial fit at location 'u_fiducial'.  The angle and associated
        parameters are written to the trace dictionary as keys beginning 'theta_'
        """
        for det_trace in det_traces:
            popt = det_trace['popt']
            pcov = det_trace['pcov']
            u_mean = det_trace['u_mean']
            gradient = Globals.polynomial(u_mean, *popt, gradient=True)
            gradient_err = np.sqrt(pcov[1][1])
            deg_rad = 180. / math.pi
            det_trace['theta'] = deg_rad * math.atan(gradient)
            det_trace['theta_err'] = gradient_err * deg_rad / (1 + gradient ** 2)
        return det_traces
