# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

refcount
#
# very innocent example
#

a = A3();
b1 = B(a);
b2 = B.create(a);


if (a.ref_count() != 3)
  error("This program will crash... now")
endif


