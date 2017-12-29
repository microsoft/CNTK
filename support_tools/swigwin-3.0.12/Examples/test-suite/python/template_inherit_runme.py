from template_inherit import *
a = FooInt()
b = FooDouble()
c = BarInt()
d = BarDouble()
e = FooUInt()
f = BarUInt()

if a.blah() != "Foo":
    raise ValueError

if b.blah() != "Foo":
    raise ValueError

if e.blah() != "Foo":
    raise ValueError

if c.blah() != "Bar":
    raise ValueError

if d.blah() != "Bar":
    raise ValueError

if f.blah() != "Bar":
    raise ValueError

if c.foomethod() != "foomethod":
    raise ValueError

if d.foomethod() != "foomethod":
    raise ValueError

if f.foomethod() != "foomethod":
    raise ValueError

if invoke_blah_int(a) != "Foo":
    raise ValueError

if invoke_blah_int(c) != "Bar":
    raise ValueError

if invoke_blah_double(b) != "Foo":
    raise ValueError

if invoke_blah_double(d) != "Bar":
    raise ValueError

if invoke_blah_uint(e) != "Foo":
    raise ValueError

if invoke_blah_uint(f) != "Bar":
    raise ValueError
