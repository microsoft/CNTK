# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

director_extend

MyObject=@() subclass(SpObject(),'getFoo',@(self) 123);
    
m = MyObject();
if (m.dummy() != 666)
  error("1st call")
endif
if (m.dummy() != 666)
  error("2nd call")
endif

