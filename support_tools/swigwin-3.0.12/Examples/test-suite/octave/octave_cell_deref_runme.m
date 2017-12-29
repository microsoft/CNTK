# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

octave_cell_deref;

assert(func("hello"));
assert(func({"hello"}));

c = func2();
assert(strcmp(c{1}, "hello"));
assert(c{2} == 4);
