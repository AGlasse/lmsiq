from lmsaiv_opt_tools import OptTools as tools
from lmsaiv_plot import Plot
from lmsaiv_as_built import AsBuilt
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
        _ = AsBuilt()
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
        method, title = OptTests.valid_test_names[test_name]
        method(test_name, title, **kwargs)
        return

    @staticmethod
    def lms_opt_01_t1(test_name, title, **kwargs):
        """ Field of view calculation using flood illuminated continuum spectral images.  Populates the slice bounds
        map in the AsBuilt object
        """
        darks = Filer.read_mosaics(inc_tags=[test_name, 'nom_dark'])
        do_plot = kwargs.get('do_plot', True)
        if do_plot:
            Plot.mosaic(darks[0], title=title)
            Plot.histograms(darks[0])
        tools.dark_stats(darks)

        floods = Filer.read_mosaics(inc_tags=[test_name, 'flood'])
        for flood in floods:
            slice_bounds, profiles = tools.flood_stats(flood)
            AsBuilt.slice_bounds = slice_bounds
            print(slice_bounds)
            n_profiles = len(profiles)
            nax_rows = int((n_profiles / 4) + 1)
            nax_cols = int(n_profiles / nax_rows)
            Plot.profiles(profiles, nax_rows=nax_rows, nax_cols=nax_cols)
            do_plot = True
            if do_plot:
                Plot.mosaic(flood, title=title, cmap='gray', sb=slice_bounds)        # Use cmap='hot', 'gray' etc.
        print('Done')
        return

    @staticmethod
    def lms_opt_01_t2(name, title):
        """ iso-alpha spectra of black body source.
        """
        for cfg_tag in ['nom', 'ext']:
            for wav_tag in ['w2700', 'w4700']:
                for a_tag in ['am200', 'ap000', 'ap200']:
                    for b_tag in ['bp000', 'bp200']:
                        obs_tags = [name, cfg_tag, wav_tag, a_tag, b_tag]
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
