from lmsaiv_opt_tests import OptTests
from lms_globals import Globals
from lms_filer import Filer


_ = Globals()
_ = Filer()
cap_name = 'lms_opt_02_dist'             # Name of analysis project
test_data_folder = 'test_toysim'
Filer.set_test_data_folder(test_data_folder)

debug_level = 'low'
Globals.set_debug_level(debug_level)
opt_tests = OptTests()
print(opt_tests)

print("Analysing data in folder data/{:s} for {:s}.".format(test_data_folder, cap_name))
opt_tests.run(cap_name, do_plot=False)
print()
print("Finished analysis of {:s}".format(cap_name))
