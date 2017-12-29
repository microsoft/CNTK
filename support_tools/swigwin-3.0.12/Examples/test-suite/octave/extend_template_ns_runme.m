# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

extend_template_ns

f = Foo_One();
if (f.test1(37) != 37)
    error
endif

if (f.test2(42) != 42)
    error
endif
