from li_std_stream import *


a = A()

o = ostringstream()

o << a << " " << 2345 << " " << 1.435


if o.str() != "A class 2345 1.435":
    print "\"%s\"" % (o.str(),)
    raise RuntimeError
