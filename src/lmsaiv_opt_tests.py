from lmsaiv_opt_tools import OptTools as tools
from lmsaiv_plot import Plot
from lms_filer import Filer


class OptTests:

    def __init__(self):
        OptTests.valid_test_names = {'lms_opt_01_t1': (OptTests.lms_opt_01_t1, 'Field of view'),
                                     'lms_opt_01_t2': (OptTests.lms_opt_01_t2, 'Iso-alpha trace, psf'),
                                     'lms_opt_01_t3': (OptTests.lms_opt_01_t3, 'Iso-lambda trace, lsf'),
                                     'lms_opt_01_t4': (OptTests.lms_opt_01_t4, 'Out-of-field straylight'),
                                     'lms_opt_01_t5': (OptTests.lms_opt_01_t5, 'Sky iso-lambda trace')
                                     }
        OptTests.data_folder = '../data/test_toysim'
        print(OptTests.valid_test_names)
        return

    @staticmethod
    def run(test_name):
        if test_name not in OptTests.valid_test_names.keys():
            print('Invalid test name, method ' + test_name + ' not found !!')
        file_list = Filer.get_file_list(OptTests.data_folder, inc_tags=[test_name])
        print('Test data found in folder ')
        [print("- {:s}".format(file)) for file in file_list]
        method, title = OptTests.valid_test_names[test_name]
        title = 'test'
        method(test_name, title)
        return

    @staticmethod
    def read_mosaics(test_name, inc_tags=[], exc_tags=[]):
        mosaics = []
        folder = OptTests.data_folder
        file_list = Filer.get_file_list(folder, inc_tags=inc_tags, exc_tags=exc_tags)
        if len(file_list) == 0:
            text = "Files in {:s} including tags ".format(folder)
            for tag in inc_tags:
                text += "{:s}, ".format(tag)
            text += 'not found'
            return mosaics

        # path = "{:s}/{:s}_{:s}.fits".format(OptTests.data_folder, test_name, obs_name)
        # hdr, data = Filer.read_fits(path, data_exts=[1, 2, 3, 4])
        for file in file_list:
            path = folder + '/' + file
            hdr, data = Filer.read_fits(path, data_exts=[1, 2, 3, 4])
            mosaic = file, hdr, data
            mosaics.append(mosaic)
        return mosaics

    @staticmethod
    def lms_opt_01_t1(test_name, title, do_plots=False):
        darks = OptTests.read_mosaics(test_name, inc_tags=['nom_dark'])
        tools.dark_stats(darks)
        if do_plots:
            tools.plot_mosaic(darks[0])

        floods = OptTests.read_mosaics(test_name, inc_tags=['flood'])
        for flood in floods:
            slice_bounds, profiles = tools.flood_stats(flood)
            Plot.profiles(profiles)
            Plot.mosaic(flood)

        print('Done')
        return

    @staticmethod
    def lms_opt_01_t2(name, title):
        print('Running ' + name + ' ' + title)
        for cfg_tag in ['nom', 'ext']:
            for wav_tag in ['w2700', 'w4700']:
                for a_tag in ['am200', 'ap000', 'ap200']:
                    for b_tag in ['bm200', 'bp000', 'bp200']:
                        obs_tags = [cfg_tag, wav_tag, a_tag, b_tag]
                        mosaics = OptTests.read_mosaics(name, inc_tags=obs_tags)
                        for mosaic in mosaics:
                            # tools.plot_mosaic(mosaic)
                            trace_data = tools.extract_alpha_trace(mosaic)

        return

    @staticmethod
    def lms_opt_01_t3(name, title):
        """ Iso-lambda images at a range of wavelengths and field positions.
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
        """ Straylight analysis.  Four images slightly beyond the bounds of the nominal and extended fields,
        followed by an image at the field centre.
        """

        return

    @staticmethod
    def lms_opt_01_t5(name, title):
        return
