# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

template_default_arg


helloInt = template_default_arg.Hello_int();
helloInt.foo(template_default_arg.Hello_int_hi);


x = template_default_arg.X_int();
if (x.meth(20.0, 200) != 200)
  error("X_int test 1 failed")
endif
if (x.meth(20) != 20)
  error("X_int test 2 failed")
endif
if (x.meth() != 0)
  error("X_int test 3 failed")
endif



y = template_default_arg.Y_unsigned();
if (y.meth(20.0, 200) != 200)
  error("Y_unsigned test 1 failed")
endif
if (y.meth(20) != 20)
  error("Y_unsigned test 2 failed")
endif
if (y.meth() != 0)
  error("Y_unsigned test 3 failed")
endif



x = template_default_arg.X_longlong();
x = template_default_arg.X_longlong(20.0);
x = template_default_arg.X_longlong(20.0, 200);


x = template_default_arg.X_int();
x = template_default_arg.X_int(20.0);
x = template_default_arg.X_int(20.0, 200);


x = template_default_arg.X_hello_unsigned();
x = template_default_arg.X_hello_unsigned(20.0);
x = template_default_arg.X_hello_unsigned(20.0, template_default_arg.Hello_int());


y = template_default_arg.Y_hello_unsigned();
y.meth(20.0, template_default_arg.Hello_int());
y.meth(template_default_arg.Hello_int());
y.meth();



fz = template_default_arg.Foo_Z_8();
x = template_default_arg.X_Foo_Z_8();
fzc = x.meth(fz);


# Templated functions

# plain function: int ott(Foo<int>)
if (template_default_arg.ott(template_default_arg.Foo_int()) != 30)
  error("ott test 1 failed")
endif

# %template(ott) ott<int, int>
if (template_default_arg.ott() != 10)
  error("ott test 2 failed")
endif
if (template_default_arg.ott(1) != 10)
  error("ott test 3 failed")
endif
if (template_default_arg.ott(1, 1) != 10)
  error("ott test 4 failed")
endif

if (template_default_arg.ott("hi") != 20)
  error("ott test 5 failed")
endif
if (template_default_arg.ott("hi", 1) != 20)
  error("ott test 6 failed")
endif
if (template_default_arg.ott("hi", 1, 1) != 20)
  error("ott test 7 failed")
endif

# %template(ott) ott<const char *>
if (template_default_arg.ottstring(template_default_arg.Hello_int(), "hi") != 40)
  error("ott test 8 failed")
endif

if (template_default_arg.ottstring(template_default_arg.Hello_int()) != 40)
  error("ott test 9 failed")
endif

# %template(ott) ott<int>
if (template_default_arg.ottint(template_default_arg.Hello_int(), 1) != 50)
  error("ott test 10 failed")
endif

if (template_default_arg.ottint(template_default_arg.Hello_int()) != 50)
  error("ott test 11 failed")
endif

# %template(ott) ott<double>
if (template_default_arg.ott(template_default_arg.Hello_int(), 1.0) != 60)
  error("ott test 12 failed")
endif

if (template_default_arg.ott(template_default_arg.Hello_int()) != 60)
  error("ott test 13 failed")
endif



