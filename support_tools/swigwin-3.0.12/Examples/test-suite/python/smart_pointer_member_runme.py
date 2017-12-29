from smart_pointer_member import *


def is_new_style_class(cls):
    return hasattr(cls, "__class__")

f = Foo()
f.y = 1

if f.y != 1:
    raise RuntimeError

b = Bar(f)
b.y = 2

if f.y != 2:
    print f.y
    print b.y
    raise RuntimeError

if b.x != f.x:
    raise RuntimeError

if b.z != f.z:
    raise RuntimeError

if is_new_style_class(Bar):  # feature not supported in old style classes
    if Foo.z == Bar.z:
        raise RuntimeError
