from nested_workaround import *

inner = Inner(5)
outer = Outer()
newInner = outer.doubleInnerValue(inner)
if newInner.getValue() != 10:
    raise RuntimeError

outer = Outer()
inner = outer.createInner(3)
newInner = outer.doubleInnerValue(inner)
if outer.getInnerValue(newInner) != 6:
    raise RuntimeError
