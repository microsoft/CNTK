# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

class_ignore

a = class_ignore.Bar();

if (!strcmp(class_ignore.do_blah(a),"Bar::blah"))
    error
endif
