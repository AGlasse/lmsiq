from lms_globals import Globals
from lms_filer import Filer
from lmsaiv_rad01 import Rad01
from lmsaiv_rad04 import Rad04
from lmsaiv_opt01 import Opt01
from lmsaiv_opt02 import Opt02
from lmsaiv_opt03 import Opt03
from lmsaiv_opt04 import Opt04
from lmsaiv_opt09 import Opt09


class OptTests:

    optical_test = None

    def __init__(self):
        """ Test dictionary, note that the key comprises 'data_source_' + 'analysis_type' (eg 'lms_opt_01_' + 'fov'.
        This acknowledges the fact that an analysis project (Jupyter notebook) may use data from several tests.
        """
        optical_test = {'lms_rad_01_dark': (Rad01.dark, 'Dark current'),
                        # 'lms_rad_03_inst_bgd': (Rad03.inst_bgd, 'Instrument background')
                        'lms_rad_04_pix_flat': (Rad04.pix_flat, 'Pixel flat field'),
                        'lms_opt_01_fov': (Opt01.fov, 'Field of view and RSRF'),
                        'lms_opt_02_dist': (Opt02.dist, 'Distortion transforms and enslitted profiles'),
                        'lms_opt_03_psf': (Opt03.psf, 'Monochromatic Point and line spread function'),
                        'lms_opt_04_ghost': (Opt04.ghost, 'Deep imaging of any ghosts detected in LMS-OPT-03'),
                        'lms_opt_09_sky': (Opt09.sky, 'Leiden sky')
                        }
        OptTests.optical_test = optical_test
        return

    def __str__(self):
        text = 'Valid analysis methods, \n'
        for cap_name in OptTests.optical_test:
            test = OptTests.optical_test[cap_name]
            text += "{:<20s},{:<20s} \n".format(cap_name, test[1])
        return text

    @staticmethod
    def get_test(cap_name, **kwargs):
        if cap_name not in OptTests.optical_test.keys():
            print()
            print("!!! Analysis project {:s} not found !!!".format(cap_name))
            quit()

        test_name = cap_name[0:10]
        file_list = Filer.get_file_list(Filer.test_data_folder, inc_tags=[test_name])
        print("Test data for {:s} found in folder ".format(test_name))
        for file in file_list:
            print("- {:s}".format(file))
        print()

        method, title = OptTests.optical_test[cap_name]
        return method, title
