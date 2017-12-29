# These examples rely on namespace packages.  Don't
# run them for old python interpreters.
import sys
import os.path

testname = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
print "Testing " + testname + " - namespace packages"

if sys.version_info < (3, 3, 0):
    print "  Not importing nstest as Python version is < 3.3"
    sys.exit(0)

import nstest

print "  Finished importing nstest"

nstest.main()
