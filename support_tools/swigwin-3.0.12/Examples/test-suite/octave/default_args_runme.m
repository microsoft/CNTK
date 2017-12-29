# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

default_args


if (default_args.Statics.staticmethod() != 60)
  error
endif

if (default_args.cfunc1(1) != 2)
  error
endif

if (default_args.cfunc2(1) != 3)
  error
endif

if (default_args.cfunc3(1) != 4)
  error
endif


f = default_args.Foo();

f.newname();
f.newname(1);


try
  f = default_args.Foo(1);
  ok = 1;
catch
  ok = 0;
end_try_catch
if (ok)
  error("Foo::Foo ignore is not working")
endif

try
  f = default_args.Foo(1,2);
  ok = 1;
catch
  ok = 0;
end_try_catch
if (ok)
  error("Foo::Foo ignore is not working")
endif

try
  f = default_args.Foo(1,2,3);
  ok = 1;
catch
  ok = 0;
end_try_catch
if (ok)
  error("Foo::Foo ignore is not working")
endif

try
  m = f.meth(1);
  ok = 1;
catch
  ok = 0;
end_try_catch
if (ok)
  error("Foo::meth ignore is not working")
endif

try
  m = f.meth(1,2);
  ok = 1;
catch
  ok = 0;
end_try_catch
if (ok)
  error("Foo::meth ignore is not working")
endif

try
  m = f.meth(1,2,3);
  ok = 1;
catch
  ok = 0;
end_try_catch
if (ok)
  error("Foo::meth ignore is not working")
endif

