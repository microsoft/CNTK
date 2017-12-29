from namespace_class import *


def is_new_style_class(cls):
    return hasattr(cls, "__class__")

try:
    p = Private1()
    error = 1
except:
    error = 0

if (error):
    raise RuntimeError, "Private1 is private"

try:
    p = Private2()
    error = 1
except:
    error = 0

if (error):
    raise RuntimeError, "Private2 is private"

if is_new_style_class(EulerT3D):
    EulerT3D.toFrame(1, 1, 1)
else:
    EulerT3D().toFrame(1, 1, 1)

b = BooT_i()
b = BooT_H()


f = FooT_i()
f.quack(1)

f = FooT_d()
f.moo(1)

f = FooT_H()
f.foo(Hi)

if is_new_style_class(FooT_H):
    f_type = str(type(f))
    if f_type.find("'namespace_class.FooT_H'") == -1:
        raise RuntimeError("Incorrect type: " + f_type)
