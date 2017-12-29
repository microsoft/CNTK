# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

using_protected

f = FooBar();
f.x = 3;

if (f.blah(4) != 4)
    error("blah(int)")
endif
