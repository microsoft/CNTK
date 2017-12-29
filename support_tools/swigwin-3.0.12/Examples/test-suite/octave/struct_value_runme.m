# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

struct_value

b = struct_value.Bar();

b.a.x = 3;
if (b.a.x != 3)
  error
endif

b.b.x = 3;
if (b.b.x != 3)
  error
endif
