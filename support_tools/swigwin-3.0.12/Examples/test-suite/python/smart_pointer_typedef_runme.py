from smart_pointer_typedef import *

f = Foo()
b = Bar(f)

b.x = 3
if b.getx() != 3:
    raise RuntimeError

fp = b.__deref__()
fp.x = 4
if fp.getx() != 4:
    raise RuntimeError
