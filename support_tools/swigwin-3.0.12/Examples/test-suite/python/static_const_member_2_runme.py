from static_const_member_2 import *

c = Test_int()
try:
    a = c.forward_field
    a = c.current_profile
    a = c.RightIndex
    a = Test_int.backward_field
    a = Test_int.LeftIndex
    a = Test_int.cavity_flags
except:
    raise RuntimeError


if Foo.BAZ.val != 2 * Foo.BAR.val:
    raise RuntimeError
