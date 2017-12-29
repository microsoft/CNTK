require("import")	-- the import fn
import("template_default_arg")	-- import code
--for k,v in pairs(template_default_arg) do _G[k]=v end -- move to global

helloInt = template_default_arg.Hello_int()
assert(template_default_arg.Hello_int_hi ~= nil)
helloInt:foo(template_default_arg.Hello_int_hi)

x = template_default_arg.X_int()
assert(x:meth(20.0, 200) == 200,"X_int test 1 failed")
assert(x:meth(20) == 20,"X_int test 2 failed")
assert(x:meth() == 0,"X_int test 3 failed")

y = template_default_arg.Y_unsigned()
assert(y:meth(20.0, 200) == 200,"Y_unsigned test 1 failed")
assert(y:meth(20) == 20,"Y_unsigned test 2 failed")
assert(y:meth() == 0,"Y_unsigned test 3 failed")

x = template_default_arg.X_longlong()
x = template_default_arg.X_longlong(20.0)
x = template_default_arg.X_longlong(20.0, 200) -- note: long longs just treated as another number

x = template_default_arg.X_int()
x = template_default_arg.X_int(20.0)
x = template_default_arg.X_int(20.0, 200)

x = template_default_arg.X_hello_unsigned()
x = template_default_arg.X_hello_unsigned(20.0)
x = template_default_arg.X_hello_unsigned(20.0, template_default_arg.Hello_int())

y = template_default_arg.Y_hello_unsigned()
y:meth(20.0, template_default_arg.Hello_int())
y:meth(template_default_arg.Hello_int())
y:meth()

fz = template_default_arg.Foo_Z_8()
x = template_default_arg.X_Foo_Z_8()
fzc = x:meth(fz)

-- Templated functions

-- plain function: int ott(Foo<int>)
assert(template_default_arg.ott(template_default_arg.Foo_int()) == 30,"ott test 1 failed")

-- %template(ott) ott<int, int>
assert(template_default_arg.ott() == 10,"ott test 2 failed")
assert(template_default_arg.ott(1) == 10,"ott test 3 failed")
assert(template_default_arg.ott(1, 1) == 10,"ott test 4 failed")

assert(template_default_arg.ott("hi") == 20,"ott test 5 failed")
assert(template_default_arg.ott("hi", 1) == 20,"ott test 6 failed")
assert(template_default_arg.ott("hi", 1, 1) == 20,"ott test 7 failed")
 
-- %template(ott) ott<const char *>
assert(template_default_arg.ottstring(template_default_arg.Hello_int(), "hi") == 40,"ott test 8 failed")
assert(template_default_arg.ottstring(template_default_arg.Hello_int()) == 40,"ott test 9 failed")

-- %template(ott) ott<int>
assert(template_default_arg.ottint(template_default_arg.Hello_int(), 1) == 50,"ott test 10 failed")
assert(template_default_arg.ottint(template_default_arg.Hello_int()) == 50,"ott test 11 failed")

-- %template(ott) ott<double>
assert(template_default_arg.ott(template_default_arg.Hello_int(), 1.0) == 60,"ott test 12 failed")
assert(template_default_arg.ott(template_default_arg.Hello_int()) == 60,"ott test 13 failed")
