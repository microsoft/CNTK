# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

smart_pointer_rename

f = Foo();
b = Bar(f);

if (b.test() != 3)
    error
endif

if (b.ftest1(1) != 1)
    error
endif

if (b.ftest2(2,3) != 2)
    error
endif

