from naturalvar import *

f = Foo()
b = Bar()

b.f = f

cvar.s = "hello"
b.s = "hello"

if b.s != cvar.s:
    raise RuntimeError
