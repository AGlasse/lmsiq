from lmsaiv_opt_tests import OptTests
from lms_globals import Globals
from lms_filer import Filer


_ = Globals()
_ = Filer()
cap_name = 'lms_opt_04_ghost'            # Name of analysis project (see lmsaiv_opt_tests.py for a full list)
test_data_folder = 'test_toysim'        # Location of test data (e.g. 'test_toysim', 'test_scope_sim' or 'test_aiv'
clear_as_built = False

Filer.set_test_data_folder(test_data_folder)

debug_level = 'low'                    # Debug information level 'off', 'low', 'medium' or 'high'
Globals.set_debug_level(debug_level)
opt_tests = OptTests()
print(opt_tests)

print("Analysing data in folder data/{:s} for {:s}.".format(test_data_folder, cap_name))
try:                                                        # Import or create new 'as built' object.
    as_built = Filer.read_pickle(Globals.as_built_file)
except FileNotFoundError:
    clear_as_built = True
if clear_as_built:
    as_built = {}

method, title = opt_tests.get_test(cap_name, do_plot=False)
as_built = method(title, as_built)
Filer.write_pickle(Globals.as_built_file, as_built)

print()
print("Finished analysis of {:s}".format(cap_name))
