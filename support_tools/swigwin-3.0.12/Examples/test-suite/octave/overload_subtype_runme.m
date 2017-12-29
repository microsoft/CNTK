# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

overload_subtype

f = Foo();
b = Bar();

if (spam(f) != 1)
    error("foo")
endif

if (spam(b) != 2)
    error("bar")
endif

