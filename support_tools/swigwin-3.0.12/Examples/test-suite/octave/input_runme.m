# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

input

f = Foo();
if (f.foo(2) != 4)
  error
endif

try
  a=f.foo();
  error
catch
end_try_catch

if (!strcmp(sfoo("Hello"),"Hello world"))
  error
endif

try
  a=sfoo();
  error
catch
end_try_catch
