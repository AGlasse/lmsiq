from lmsaiv_opt_tests import OptTests
from lms_globals import Globals
from lms_filer import Filer


_ = Globals()
_ = Filer()
opt_tests = OptTests()
print(opt_tests)

cap_name = 'lms_opt_01_fov'             # Name of analysis project
debug_level = 'low'

print("Analysing test data for {:s}.".format(cap_name))
Globals.set_debug_level(debug_level)
opt_tests.run(cap_name, do_plot=False)
print()
print("Finished analysis of {:s}".format(cap_name))
