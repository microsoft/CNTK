import voidtest

voidtest.globalfunc()
f = voidtest.Foo()
f.memberfunc()

voidtest.Foo_staticmemberfunc()


def fvoid():
    pass

if f.memberfunc() != fvoid():
    raise RuntimeError


v1 = voidtest.vfunc1(f)
v2 = voidtest.vfunc2(f)
if v1 != v2:
    raise RuntimeError

v3 = voidtest.vfunc3(v1)
if v3.this != f.this:
    raise RuntimeError
v4 = voidtest.vfunc1(f)
if v4 != v1:
    raise RuntimeError


v3.memberfunc()
