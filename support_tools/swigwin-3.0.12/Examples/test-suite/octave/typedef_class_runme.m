# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

typedef_class

a = typedef_class.RealA();
a.a = 3;
 
b = typedef_class.B();
b.testA(a);
