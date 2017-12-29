# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

mod_a
mod_b

c = mod_b.C();
d = mod_b.D();
d.DoSomething(c);

