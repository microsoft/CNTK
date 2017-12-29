from using_private import *

f = FooBar()
f.x = 3

if f.blah(4) != 4:
    raise RuntimeError, "blah(int)"

if f.defaulted() != -1:
    raise RuntimeError, "defaulted()"

if f.defaulted(222) != 222:
    raise RuntimeError, "defaulted(222)"
