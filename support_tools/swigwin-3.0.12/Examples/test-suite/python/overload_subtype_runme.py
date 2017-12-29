from overload_subtype import *

f = Foo()
b = Bar()

if spam(f) != 1:
    raise RuntimeError, "foo"

if spam(b) != 2:
    raise RuntimeError, "bar"
