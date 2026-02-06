from lms_globals import Globals
from lms_filer import Filer
from lmsaiv_opt01 import Opt01
from lmsaiv_opt02 import Opt02
from lmsaiv_opt03 import Opt03


class OptTests:

    optical_test = None

    def __init__(self):
        Filer.set_test_data_folder('scopesim')
        # Test dictionary, note that the key comprises 'data_source_' + 'analysis' (eg 'lms_opt_01_' + 'fov'
        optical_test = {'lms_opt_01_fov': (Opt01.fov, 'Field of view and RSRF'),
                        'lms_opt_02_dist': (Opt02.dist, 'Distortion transforms and enslitted profiles'),
                        'lms_opt_03_psf': (Opt03.psf, 'Monochomatic Point and line spread function'),
                        # 'lms_opt_08': (OptTests.lms_opt_08, 'Out-of-field straylight'),
                        }
        OptTests.optical_test = optical_test
        return

    def __str__(self):
        text = 'Valid analysis methods, \n'
        for key in OptTests.optical_test:
            test = OptTests.optical_test[key]
            text += "{:<20s},{:<20s} \n".format(key, test[1])
        return text

    @staticmethod
    def run(cap_name, **kwargs):
        if cap_name not in OptTests.optical_test.keys():
            print("Test {:s} not found !!!".format(cap_name))
            return
        file_list = Filer.get_file_list(Filer.test_data_folder, inc_tags=[cap_name])
        print('Test data found in folder ')
        [print("- {:s}".format(file)) for file in file_list]
        print()
        try:            # Import or create 'as built' object.
            as_built = Filer.read_pickle(Globals.as_built_file)
        except FileNotFoundError:
            as_built = {}
        method, title = OptTests.optical_test[cap_name]
        as_built = method(title, as_built, **kwargs)
        Filer.write_pickle(Globals.as_built_file, as_built)
        return
