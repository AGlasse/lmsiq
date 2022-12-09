import numpy as np
import math


from photutils import centroid_com
from photutils.aperture import RectangularAperture
from lmsiq_filer import LMSIQFiler as filer
from lms_globals import Globals


class LMSIQAnalyse:

    def __init__(self):
        return

    @staticmethod
    def load_observations(path, file_list, run_test):
        """ Load image and parameter information as a list of observation objects from a list of files, sorting them
        in order 'perfect, design then obs_id = 00, 01 etc.
        :param path:
        :param file_list:
        :param run_test: Boolean.  True = Replace image data with a single bright detector pixel at the image centre.
        :return:
        """
        n_obs = len(file_list)
        observations = [None]*n_obs
        for file in file_list:
            file_id = file[0:-5].split('_')[4]
            if file_id == 'perfect':
                j = 0
            else:
                if file_id == 'design':
                    j = 1
                else:
                    j = int(file_id) + 2
            image, header = filer.read_zemax_fits(path, file)
            mm_fitspix = 1000.0 * header['CDELT1']
            if run_test:
                oversampling = int((Globals.mm_lmspix / mm_fitspix) + 0.5)
                image[:, :] = 0.0
                nr, nc = image.shape
                r1, c1 = int((nr - oversampling) / 2.0), int((nc - oversampling) / 2.0)
                r2, c2 = r1 + oversampling, c1 + oversampling
                image[r1:r2, c1:c2] = 1.0



            params = file_id, mm_fitspix
            observations[j] = image, params
        return observations

    @staticmethod
    def find_mean_centroid(observations):
        """ Find the mean centroid position for all images in a list.
        """
        images = []
        for obs in observations:
            image, _ = obs
            images.append(image)
        cube = np.array(images)
        image_ave = np.average(cube, axis=0)
        centroid = centroid_com(image_ave)
        return centroid

    @staticmethod
    def eed(observations, axis, **kwargs):
        """ Calculate the enslitted energy along an axis.
        :param observations:    List of image, header pairs
        :param axis: Variable axis, 'spectral' or 'spatial'
        :param kwargs: is_log = True for samples which are uniform in log space
        :return radii: Sampled axis
                ees_mean: Mean enslitted energy profile
                ees_rms:
                ees_all: EE profile averaged for all images
        """
        debug = kwargs.get('debug', False)
        is_log = kwargs.get('log10sampling', True)  # Use sampling equispaced in log10

        n_obs = len(observations)
        r_sample = 0.1
        r_start = r_sample
        r_max = 60.0   # Maximum radial size of aperture (a bit less than 1/2 image size)

        radii = np.arange(r_start, r_max, r_sample)
        n_points = radii.shape[0]
        if is_log:      # Remap the same number of points onto a log uniform scale.
            k = math.log10(r_max / r_start) / (n_points - 1)
            lr = math.log10(r_start)
            for i in range(0, n_points):
                radii[i] = math.pow(10., lr)
                lr += k

        ees_all = np.zeros((n_points, n_obs))
        im_sums = np.zeros(n_obs)
        im_peaks = np.zeros(n_obs)

        for j, obs in enumerate(observations):
            image, params = obs
            file_id, mm_fitspix = params
            imin, imax = np.amin(image), np.amax(image)
            centroid = centroid_com(image)
            if debug:
                print('Processing file {:s} into column {:d} {:10.1e} {:10.1e}'.format(file_id, j, imin, imax))
            for i in range(0, n_points):        # One radial point per row
                r = radii[i]      # Increase aperture width to measure spectral cog profile
                if axis == 'spectral':
                    aperture = RectangularAperture(centroid, w=2.*r, h=2.*r_max)     # Spectral
                if axis == 'spatial':
                    aperture = RectangularAperture(centroid, w=2.*r_max, h=2.*r)     # Spatial
                ees_all[i, j] = LMSIQAnalyse.exact_rectangular(image, aperture)

            im_sums[j] = ees_all[n_points-1, j]
            im_peaks[j] = imax

        ees_mean = np.zeros(n_points)
        ees_rms = np.zeros(n_points)
        for j in range(0, n_obs):
            enorm = ees_all[n_points-1, j]
            ees_all[:, j] = np.divide(ees_all[:, j], enorm)
        for i in range(0, n_points):
            ees_mean[i] = np.mean(ees_all[i, :])
            ees_rms[i] = np.std(ees_all[i, :])
        # Rescale x axis from image scale to LMS pixels
        lmspix_fitspix = mm_fitspix / Globals.mm_lmspix
        xlms = np.multiply(radii, lmspix_fitspix)
        ees_data = xlms, ees_mean, ees_rms, ees_all
        return ees_data

    @staticmethod
    def strehl(observations):
        """ Calculate the Strehl ratio as the ratio between the peak amplitude of the mean image and
        the peak amplitude of the perfect image, where both images have been normalised to have a total
        signal of unity
        """

        perfect_image, _ = observations[0]
        perfect_power = np.sum(perfect_image)
        perfect_peak = np.amax(perfect_image)
        # Calculate the error on the Strehl from the individual images
        strehl_list = []
        for obs in observations:
            image, params = obs
            power = np.sum(image)
            peak = np.amax(image)
            strehl = (peak * perfect_power) / (perfect_peak * power)
            strehl_list.append(strehl)
        strehl_err = np.std(np.array(strehl_list))
        strehl_mean = np.mean(np.array(strehl_list))
        return strehl_mean, strehl_err

    @staticmethod
    def lsf(observations, axis, **kwargs):
        """ Find the line spread function for all image files.
        :returns - uvals, pixel scale
                 - lsf_mean, mean line spread function for all images
                 - lsf_rms, root mean square distribution per pixel
                 - lsf, array of line spread functions for each image
        """
        debug = kwargs.get('debug', False)

        centroid = LMSIQAnalyse.find_mean_centroid(observations)

        v_coadd = 12.0       # Number of image pixels to coadd orthogonal to profile
        n_files = len(observations)

        u_sample = 1.0      # Sample psf once per pixel to avoid steps
        u_start = 0.0       # Offset from centroid
        u_radius = 20.0     # Maximum radial size of aperture (a bit less than 1/2 image size)

        uvals = np.arange(u_start - u_radius, u_start + u_radius, u_sample)
        n_points = uvals.shape[0]
        lsf_all = np.zeros((n_points, n_files))
        lsf_mean = np.zeros(n_points)
        lsf_rms = np.zeros(n_points)

        for j, obs in enumerate(observations):
            image, params = obs
            file_id, mm_fitspix = params
            if debug:
                print('Processing file {:s} into column {:d}'.format(file_id, j))
            ap_pos = np.array(centroid)
            u_cen = centroid[0] if axis == 'spectral' else centroid[1]
            us = np.add(uvals, u_cen)
            for i in range(0, n_points):  # One radial point per row
                u = us[i]
                if axis == 'spectral':
                    ap_pos[0] = u
                    aperture = RectangularAperture(ap_pos, w=u_sample, h=v_coadd)  # Spectral
                    lsf_all[i, j] = LMSIQAnalyse.exact_rectangular(image, aperture)

                if axis == 'spatial':
                    ap_pos[1] = u
                    aperture = RectangularAperture(ap_pos, w=v_coadd, h=u_sample)  # Spatial
                    lsf_all[i, j] = LMSIQAnalyse.exact_rectangular(image, aperture)
        for i in range(0, n_points):    # Find mean of Monte-Carlo spectra
            lsf_mean[i] = np.mean(lsf_all[i, 2:])
            lsf_rms[i] = np.std(lsf_all[i, 2:])
        # Convert x scale to LMS pixels
        lmspix_fitspix = mm_fitspix / Globals.mm_lmspix
        xlms = np.multiply(uvals, lmspix_fitspix)           # Convert scale to LMS pixels
        return xlms, lsf_mean, lsf_rms, lsf_all

    @staticmethod
    def find_ee_axis_references(wav, axis, xlms, lsf_mean, lsf_all):
        """ Calculate the EED for the three EE profiles (mean, perfect and design) at a reference aperture. """
        ees = [lsf_mean, lsf_all[:,0], lsf_all[:,1]]
        x_ref_min = 2.0 if axis == 'spectral' else 3.0
        broadening = 1.0 if wav < 3.7 else wav / 3.7
        x_ref = x_ref_min * broadening
        ee_refs = []
        for ee in ees:
            iz = np.where(xlms > x_ref)
            i = iz[0][0]
            ee_ref = ee[i - 1] + (x_ref - xlms[i - 1]) * (ee[i] - ee[i - 1]) / (xlms[i] - xlms[i - 1])
            ee_refs.append(ee_ref)
        return x_ref, ee_refs

    @staticmethod
    def exact_rectangular(image, aperture):
        cen = aperture.positions
        w = aperture.w
        h = aperture.h
        x1 = cen[0] - w / 2.0
        x2 = x1 + w
        y1 = cen[1] - h / 2.0
        y2 = y1 + h
        c1, c2, r1, r2 = int(x1), int(x2), int(y1), int(y2)
        nr = r2 - r1 + 1      # Number of rows in subarray, 1 pixel extra to allow sub-pixel fragments.
        nc = c2 - c1 + 1
        wts = np.ones((nr, nc))
        im = image[r1:r1+nr, c1:c1+nc]

        fc1 = 1. - (x1 - c1)
        fc2 = x2 - c2
        if nc == 1:
            fc = fc1 + fc2 - 1.
            wts[:,0] *= fc
        else:
            wts[:,0] *= fc1
            wts[:,nc-1] *= fc2

        fr1 = 1. - (y1 - r1)
        fr2 = y2 - r2
        if nr == 1:
            fr = fr1 + fr2 - 1.
            wts[0,:] *= fr
        else:
            wts[0,:] *= fr1
            wts[nr-1,:] *= fr2

        wtim = np.multiply(im, wts)
        f = np.sum(wtim)
        return f
