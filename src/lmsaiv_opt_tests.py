import numpy as np
from astropy import units as u
from lms_globals import Globals
from lmsaiv_opt_tools import OptTools
from lmsaiv_plot import Plot
from lmssim_model import Model
from lms_filer import Filer


class OptTests:


    def __init__(self):
        OptTests.valid_test_names = {'lms_opt_01_t1': (OptTests.lms_opt_01_t1, 'Field of view'),
                                     'lms_opt_01_t2': (OptTests.lms_opt_01_t2, 'Grey body iso-alpha, along slice PSF'),
                                     'lms_opt_01_t3': (OptTests.lms_opt_01_t3, 'Laser iso-lambda, LSF'),
                                     'lms_opt_01_t4': (OptTests.lms_opt_01_t4, 'Sky iso-lambda trace'),
                                     'lms_opt_08': (OptTests.lms_opt_08, 'Out-of-field straylight'),
                                     }
        Filer.set_test_data_folder('scopesim')
        return

    def __str__(self):
        text = 'OptTests available for tests.. \n'
        for test_name in self.valid_test_names:
            test = self.valid_test_names[test_name]
            text += "{:<20s},{:<20s} \n".format(test_name, test[1])
        return text

    @staticmethod
    def run(test_name, **kwargs):
        if test_name not in OptTests.valid_test_names.keys():
            print('Invalid test name, method ' + test_name + ' not found !!')
        file_list = Filer.get_file_list(Filer.test_data_folder, inc_tags=[test_name])
        print('Test data found in folder ')
        [print("- {:s}".format(file)) for file in file_list]
        try:            # Check that 'as built' dictionary file exists
            as_built = Filer.read_pickle(Globals.as_built_file)       # , 'rb')
        except:
            as_built = {}
        method, title = OptTests.valid_test_names[test_name]
        as_built = method(test_name, title, as_built, **kwargs)
        Filer.write_pickle(Globals.as_built_file, as_built)
        return

    @staticmethod
    def lms_opt_01_t1(test_name, title, as_built, **kwargs):
        """ Field of view calculation using flood illuminated continuum spectral images.  Populates the slice bounds
        map in the AsBuilt object
        """
        darks = Filer.read_mosaics(inc_tags=[test_name, 'nom_dark'])
        do_plot = kwargs.get('do_plot', True)
        if do_plot:
            Plot.mosaic(darks[0], title=title)
            Plot.histograms(darks[0])
        OptTools.dark_stats(darks)

        floods = Filer.read_mosaics(inc_tags=[test_name, 'flood'])
        for flood in floods:
            slice_map, profiles = OptTools.flood_stats(flood)
            do_plot = False
            if do_plot:
                Plot.profiles(profiles)
                Plot.mosaic(flood, title=title, cmap='hot')        # Use cmap='hot', 'gray' etc.
                Plot.mosaic(slice_map, title='Slice Map', cmap='hsv', mask=(0.0, 'black'))
            as_built['slice_map'] = slice_map

        # Generate relative response tuple.
        cols = np.arange(0, 4096, 1)
        for flood in floods:
            slice_map = as_built['slice_map']
            rrf = OptTools.copy_mosaic(slice_map, copy_name='rel_res_function')
            rrf_name, rrf_primary_header, rrf_hdus = rrf
            Plot.mosaic(slice_map, title='Slice Map', cmap='hsv', mask=(0.0, 'black'))
            name, primary_hdr, hdus = flood
            wave_mosaic_cen = primary_hdr['HIERARCH ESO INS WLEN CEN'] * u.micron
            _, _, slice_map_hdus = slice_map
            for i in range(0, 4):
                slice_map_data = slice_map_hdus[i].data
                slice_mask = np.where(slice_map_data > 0., 1., 0.)
                # Very approximate dispersion...!
                hdr = hdus[i].header
                flood_image = hdus[i].data
                x_det_cen = float(hdr['X_CEN']) * u.mm
                n_det_cols = float(hdr['X_SIZE'])
                pix_size = hdr['HIERARCH pixel_size'] * u.mm
                c_det_cen = x_det_cen / pix_size
                c_det_org = c_det_cen - n_det_cols / 2
                disp = .08 * u.micron / (2. * n_det_cols)
                waves = wave_mosaic_cen + disp * (c_det_org + cols)
                flux = Model.black_body(waves, tbb=1000.)
                n_det_rows = int(hdr['Y_SIZE'])
                rrf_image = rrf_hdus[i].data
                for row in range(0, n_det_rows):
                    idx = np.argwhere(slice_mask[row] > 0.)
                    rrf_image[row, idx] = flood_image[row, idx] / flux[idx]
                rrf_hdus[i].data = rrf_image
            Plot.mosaic(rrf, title='Rel Response Function', cmap='grey', mask=(0.0, 'black'))
        print('Done')
        return as_built

    @staticmethod
    def lms_opt_01_t2(test_name, title, **kwargs):
        """ iso-alpha spectra of black body source.
        """
        # Start by processing all grid
        mosaics = Filer.read_mosaics(inc_tags=[test_name, 'grid'])
        for mosaic in mosaics:
            file_name, hdr, hdus = mosaic
            # For now we get the wavelength and field position from the file name.  Header later.
            wave_tag = file_name[28:32]
            wave = float(wave_tag) / 1000.      # Laser wavelength in microns
            alpha_sign = 1. if file_name[34] == 'p' else -1.
            alpha_tag = file_name[35:38]
            beta_sign = 1. if file_name[40] == 'p' else -1.
            beta_tag = file_name[41:44]
            alpha = alpha_sign * float(alpha_tag)           # alpha/beta coordinates in mas.
            beta = beta_sign * float(beta_tag)

        for cfg_tag in ['nom', 'ext']:
            for wav_tag in ['w2700', 'w4700']:
                for a_tag in ['am350', 'ap000', 'ap350']:
                    for b_tag in ['bp000', 'bp200']:
                        obs_tags = [test_name, cfg_tag, wav_tag, a_tag, b_tag]
                        mosaics = Filer.read_mosaics(inc_tags=obs_tags)
                        for mosaic in mosaics:
                            profiles = tools.extract_alpha_traces(mosaic)
                            Plot.profiles(profiles)
                            Plot.mosaic(mosaic)

        return

    @staticmethod
    def lms_opt_01_t3(name, title):
        """ Laser generated iso-lambda images at a range of wavelengths and field positions.
        Extract efp_x, efp_y, efp_w data for transform calculations
        """
        print('Running ' + name + ' ' + title)
        for cfg_tag in ['nom', 'ext']:
            for wav_tag in ['w2700', 'w4700']:
                for a_tag in ['am200', 'ap000', 'ap200']:
                    for b_tag in ['bm200', 'bp000', 'bp200']:
                        obs_tags = [cfg_tag, wav_tag, a_tag, b_tag]
                        mosaics = OptTests.read_mosaics(name, inc_tags=obs_tags)
                        for mosaic in mosaics:
                            tools.plot_mosaic(mosaic)

        return

    @staticmethod
    def lms_opt_01_t4(name, title):
        """ Leiden sky spectral analysis.
        """

        return

    @staticmethod
    def lms_opt_08(name, title):
        return
