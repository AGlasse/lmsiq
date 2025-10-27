from lmsaiv_opt_tests import OptTests
import lms_globals as Globals


_ = Globals
opt_tests = OptTests()
print(opt_tests)

test_name = 'lms_opt_01_t2'
fmt = "lmsopt - Analysing test data for {:s} - {:s}"
print(fmt.format(test_name, 'started'))
print()
opt_tests.run(test_name, do_plot=False)
print()
print(fmt.format(test_name, 'done'))
