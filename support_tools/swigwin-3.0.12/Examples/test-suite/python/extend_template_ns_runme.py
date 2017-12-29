from extend_template_ns import *
f = Foo_One()
if f.test1(37) != 37:
    raise RuntimeError

if f.test2(42) != 42:
    raise RuntimeError
