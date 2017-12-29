# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

template_ns4

d = make_Class_DD();
if (!strcmp(d.test(),"test"))
  error
endif
