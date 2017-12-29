# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

using2

if (using2.spam(37) != 37)
    error
endif
