from smart_pointer_rename import *

f = Foo()
b = Bar(f)

if b.test() != 3:
    raise RuntimeError

if b.ftest1(1) != 1:
    raise RuntimeError

if b.ftest2(2, 3) != 2:
    raise RuntimeError
