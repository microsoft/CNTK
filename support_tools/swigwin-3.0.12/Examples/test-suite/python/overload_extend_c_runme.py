import overload_extend_c

f = overload_extend_c.Foo()
if f.test() != 0:
    raise RuntimeError
if f.test(3) != 1:
    raise RuntimeError
if f.test("hello") != 2:
    raise RuntimeError
if f.test(3, 2) != 5:
    raise RuntimeError
if f.test(3.0) != 1003:
    raise RuntimeError
