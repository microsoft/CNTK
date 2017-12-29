import template_default_arg


helloInt = template_default_arg.Hello_int()
helloInt.foo(template_default_arg.Hello_int.hi)


x = template_default_arg.X_int()
if (x.meth(20.0, 200) != 200):
    raise RuntimeError, ("X_int test 1 failed")
if (x.meth(20) != 20):
    raise RuntimeError, ("X_int test 2 failed")
if (x.meth() != 0):
    raise RuntimeError, ("X_int test 3 failed")


y = template_default_arg.Y_unsigned()
if (y.meth(20.0, 200) != 200):
    raise RuntimeError, ("Y_unsigned test 1 failed")
if (y.meth(20) != 20):
    raise RuntimeError, ("Y_unsigned test 2 failed")
if (y.meth() != 0):
    raise RuntimeError, ("Y_unsigned test 3 failed")


x = template_default_arg.X_longlong()
x = template_default_arg.X_longlong(20.0)
x = template_default_arg.X_longlong(20.0, 200L)


x = template_default_arg.X_int()
x = template_default_arg.X_int(20.0)
x = template_default_arg.X_int(20.0, 200)


x = template_default_arg.X_hello_unsigned()
x = template_default_arg.X_hello_unsigned(20.0)
x = template_default_arg.X_hello_unsigned(
    20.0, template_default_arg.Hello_int())


y = template_default_arg.Y_hello_unsigned()
y.meth(20.0, template_default_arg.Hello_int())
y.meth(template_default_arg.Hello_int())
y.meth()


fz = template_default_arg.Foo_Z_8()
x = template_default_arg.X_Foo_Z_8()
fzc = x.meth(fz)


# Templated functions

# plain function: int ott(Foo<int>)
if (template_default_arg.ott(template_default_arg.Foo_int()) != 30):
    raise RuntimeError, ("ott test 1 failed")

# %template(ott) ott<int, int>
if (template_default_arg.ott() != 10):
    raise RuntimeError, ("ott test 2 failed")
if (template_default_arg.ott(1) != 10):
    raise RuntimeError, ("ott test 3 failed")
if (template_default_arg.ott(1, 1) != 10):
    raise RuntimeError, ("ott test 4 failed")

if (template_default_arg.ott("hi") != 20):
    raise RuntimeError, ("ott test 5 failed")
if (template_default_arg.ott("hi", 1) != 20):
    raise RuntimeError, ("ott test 6 failed")
if (template_default_arg.ott("hi", 1, 1) != 20):
    raise RuntimeError, ("ott test 7 failed")

# %template(ott) ott<const char *>
if (template_default_arg.ottstring(template_default_arg.Hello_int(), "hi") != 40):
    raise RuntimeError, ("ott test 8 failed")

if (template_default_arg.ottstring(template_default_arg.Hello_int()) != 40):
    raise RuntimeError, ("ott test 9 failed")

# %template(ott) ott<int>
if (template_default_arg.ottint(template_default_arg.Hello_int(), 1) != 50):
    raise RuntimeError, ("ott test 10 failed")

if (template_default_arg.ottint(template_default_arg.Hello_int()) != 50):
    raise RuntimeError, ("ott test 11 failed")

# %template(ott) ott<double>
if (template_default_arg.ott(template_default_arg.Hello_int(), 1.0) != 60):
    raise RuntimeError, ("ott test 12 failed")

if (template_default_arg.ott(template_default_arg.Hello_int()) != 60):
    raise RuntimeError, ("ott test 13 failed")
