from smart_pointer_overload import *

f = Foo()
b = Bar(f)


if f.test(3) != 1:
    raise RuntimeError
if f.test(3.5) != 2:
    raise RuntimeError
if f.test("hello") != 3:
    raise RuntimeError

if b.test(3) != 1:
    raise RuntimeError
if b.test(3.5) != 2:
    raise RuntimeError
if b.test("hello") != 3:
    raise RuntimeError
