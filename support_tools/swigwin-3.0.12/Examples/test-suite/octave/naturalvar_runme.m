# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

naturalvar

f = Foo();
b = Bar();

b.f = f;

cvar.s = "hello";
b.s = "hello";

if (b.s != cvar.s)
    error
endif

