# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

li_carrays_cpp

d = doubleArray(10);

d(0) = 7;
d(5) = d(0) + 3;

if (d(5) + d(0) != 17)
    error
endif
