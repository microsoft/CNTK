require("import")	-- the import fn
import("dynamic_cast")	-- import code

f = dynamic_cast.Foo()
b = dynamic_cast.Bar()

x = f:blah()
y = b:blah()

-- swig_type is a swiglua specific function which gets the swig_type_info's name
assert(swig_type(f)==swig_type(x))
assert(swig_type(b)==swig_type(y))

-- the real test: is y a Foo* or a Bar*?
assert(dynamic_cast.do_test(y)=="Bar::test")
