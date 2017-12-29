# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

sneaky1
x = sneaky1.add(3,4);
y = sneaky1.subtract(3,4);
z = sneaky1.mul(3,4);
w = sneaky1.divide(3,4);
