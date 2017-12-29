# This is the import runtime testcase.

import imports_b
import imports_a
import sys

x = imports_b.B()
imports_a.A.hello(x)

a = imports_a.A()

c = imports_b.C()
a1 = c.get_a(c)
a2 = c.get_a_type(c)

if a1.hello() != a2.hello():
    raise RuntimeError
