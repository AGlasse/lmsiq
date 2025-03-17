from lmsaiv_opt_tests import OptTests
from lmsaiv_as_built import AsBuilt


opt_tests = OptTests()
test_name = 'lms_opt_01_t1'

fmt = "lmsopt - Analysing test data for {:s} - {:s}"
print(fmt.format(test_name, 'started'))
print()
opt_tests.run(test_name)
print()
print(fmt.format(test_name, 'done'))
