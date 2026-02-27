from lmsaiv_opt_tests import OptTests
from lms_globals import Globals
from lms_filer import Filer


_ = Globals()
_ = Filer()

test_data_folder = 'test_toysim'
opt_tests = OptTests(test_data_folder)
print(opt_tests)

cap_name = 'lms_opt_01_fov'             # Name of analysis project
debug_level = 'low'

print("Analysing data in folder data/{:s} for {:s}.".format(test_data_folder, cap_name))
Globals.set_debug_level(debug_level)
opt_tests.run(cap_name, do_plot=False)
print()
print("Finished analysis of {:s}".format(cap_name))
