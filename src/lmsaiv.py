from lmsaiv_opt_tests import OptTests
from lms_globals import Globals


_ = Globals()
opt_tests = OptTests()
print(opt_tests)

test_name = 'lms_opt_02'
debug_level = 'low'
Globals.debug_level = debug_level

print("Analysing test data for {:s}.".format(test_name))
debug_levels = list(Globals.debug_levels.keys())
debug = Globals.debug_levels[debug_level]
debug_text = "Debug level set to '{:s}'.  ['{:s}'".format(debug_level, debug_levels[0])
for dlt in debug_levels[1:]:
    debug_text += ", '{:s}'".format(dlt)
print(debug_text + ']')
print()
opt_tests.run(test_name, do_plot=False)
print()
print("Finished analysis of {:s}".format(test_name))
