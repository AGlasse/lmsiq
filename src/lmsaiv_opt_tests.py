from lmsaiv_opt_tools import OptTools as tools
from lms_filer import Filer


class OptTests:

    def __init__(self):
        OptTests.valid_test_names = {'lms_opt_01_t1': (OptTests.lms_opt_01_t1, 'Field of view'),
                                     'lms_opt_01_t2': (OptTests.lms_opt_01_t2, 'Iso-alpha trace'),
                                     'lms_opt_01_t3': (OptTests.lms_opt_01_t3, 'Iso-lambda trace')
                                     }
        OptTests.data_folder = '../data/test_sim'
        print(OptTests.valid_test_names)
        return

    @staticmethod
    def run(test_name):
        if test_name not in OptTests.valid_test_names.keys():
            print('Invalid test name, method ' + test_name + ' not found !!')
        file_list = Filer.get_file_list(OptTests.data_folder, inc_tags=[test_name])
        print('Test data found in folder ')
        [print("- {:s}".format(file)) for file in file_list]
        # method = OptTests.test_name
        method, title = OptTests.valid_test_names[test_name]
        title = 'test'
        method(test_name, title)
        return

    @staticmethod
    def read_mosaic(test_name, obs_name):
        path = "{:s}/{:s}_{:s}.fits".format(OptTests.data_folder, test_name, obs_name)
        hdr, data = Filer.read_fits(path, data_exts=[1, 2, 3, 4])
        mosaic = obs_name, hdr, data
        return mosaic

    @staticmethod
    def lms_opt_01_t1(test_name, title):
        dark = OptTests.read_mosaic(test_name, 'nom_dark1')
        tools.dark_stats(dark)
        tools.plot_mosaic(dark)

        flood = OptTests.read_mosaic(test_name, 'nom_flood')
        tools.flood_stats(flood)
        tools.plot_mosaic(flood)
        print('Done')
        return

    @staticmethod
    def lms_opt_01_t2(name, title):
        print('Running ' + name + ' ' + title)
        mosaic = OptTests.read_mosaic(name, 'nom_cfopnh')
        tools.plot_mosaic(mosaic)
        return

    @staticmethod
    def lms_opt_01_t3(name, title):
        print('Running ' + name + ' ' + title)
        mosaic = OptTests.read_mosaic(name, 'nom_lt_w4700')
        tools.plot_mosaic(mosaic)
        return
