import overload_extend2

f = overload_extend2.Foo()
if f.test(3) != 1:
    raise RuntimeError
if f.test("hello") != 2:
    raise RuntimeError
if f.test(3.5, 2.5) != 3:
    raise RuntimeError
if f.test("hello", 20) != 1020:
    raise RuntimeError
if f.test("hello", 20, 100) != 120:
    raise RuntimeError

# C default args
if f.test(f) != 30:
    raise RuntimeError
if f.test(f, 100) != 120:
    raise RuntimeError
if f.test(f, 100, 200) != 300:
    raise RuntimeError
