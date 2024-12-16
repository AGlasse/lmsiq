from lms_filer import Filer
from lmsopt_analysis import OptAnalysis


test_name = 'lms_opt_01_t1'
fmt = "lmsopt - Analysis of test data {:s} - {:s}"
print(fmt.format(test_name, 'started'))
print()
lmsopt_analysis = OptAnalysis()
lmsopt_analysis.run(test_name)
print()
print(fmt.format(test_name, 'done'))
