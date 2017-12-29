-- demo of lua swig
require("import")	-- the import fn
import("voidtest")	-- import lib

-- test calling functions
voidtest.globalfunc()
f = voidtest.Foo()
f:memberfunc()  -- member fns must have : not a .

voidtest.Foo_staticmemberfunc() -- static member fns are still a little messy

v1 = voidtest.vfunc1(f)
v2 = voidtest.vfunc2(f)

assert(swig_equals(v1,v2)) -- a raw equals will not work, we look at the raw pointers

v3 = voidtest.vfunc3(v1)
assert(swig_equals(v3,f))

v4 = voidtest.vfunc1(f)
assert(swig_equals(v4,v1))

v3:memberfunc()

-- also testing nil's support
-- nil, are acceptable anywhere a pointer is
n1 = voidtest.vfunc1(nil)
n2 = voidtest.vfunc2(nil)

assert(n1==nil)
assert(n2==nil)

n3 = voidtest.vfunc3(n1)
n4 = voidtest.vfunc1(nil)

assert(n3==nil)
assert(n4==nil)
