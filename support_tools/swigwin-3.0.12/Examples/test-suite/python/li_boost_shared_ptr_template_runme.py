from li_boost_shared_ptr_template import *

b = BaseINTEGER()
d = DerivedINTEGER()
if b.bar() != 1:
    raise RuntimeError
if d.bar() != 2:
    raise RuntimeError
if bar_getter(b) != 1:
    raise RuntimeError
if bar_getter(d) != 2:
    raise RuntimeError

b = BaseDefaultInt()
d = DerivedDefaultInt()
d2 = DerivedDefaultInt2()
if b.bar2() != 3:
    raise RuntimeError
if d.bar2() != 4:
    raise RuntimeError
if d2.bar2() != 4:
    raise RuntimeError
if bar2_getter(b) != 3:
    raise RuntimeError
# SWIG fix reverted in Subversion rev 12953
# Testcase has now been modified to mask the problem by providing the default parameter 'int' in:
#   %shared_ptr(Space::BaseDefault<short, int>)
# If this is not done then d fails to convert to BaseDefault<short>&
if bar2_getter(d) != 4:
    raise RuntimeError
if bar2_getter(d2) != 4:
    raise RuntimeError
