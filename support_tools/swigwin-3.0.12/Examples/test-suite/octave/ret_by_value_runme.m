# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

ret_by_value

a = ret_by_value.get_test();
if (a.myInt != 100)
    error
endif

if (a.myShort != 200)
    error
endif
