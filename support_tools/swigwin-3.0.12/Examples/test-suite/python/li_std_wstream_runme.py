from li_std_wstream import *


a = A()

o = wostringstream()

o << a << u" " << 2345 << u" " << 1.435 << wends

if o.str() != "A class 2345 1.435\0":
    print "\"%s\"" % (o.str(),)
    raise RuntimeError
