from li_boost_shared_ptr_bits import *


def is_new_style_class(cls):
    return hasattr(cls, "__class__")


def check(nd):
    nd.i = 200
    i = nd.i

    try:
        nd.notexist = 100
        passed = 0
    except:
        passed = 1

    if not passed:
        raise "Test failed"

nd = NonDynamic()
check(nd)
b = boing(nd)
check(b)

################################

v = VectorIntHolder()
v.push_back(IntHolder(11))
v.push_back(IntHolder(22))
v.push_back(IntHolder(33))

sum = sum(v)
if sum != 66:
    raise "sum is wrong"

################################
if is_new_style_class(HiddenDestructor):
    p = HiddenDestructor.create()
else:
    p = HiddenDestructor_create()
