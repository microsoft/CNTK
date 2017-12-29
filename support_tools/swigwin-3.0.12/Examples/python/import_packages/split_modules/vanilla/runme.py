import os.path

testname = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
print "Testing " + testname + " - split modules"

import pkg1.foo

print "  Finished importing pkg1.foo"

assert(pkg1.foo.count() == 3)
