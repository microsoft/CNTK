# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

using_composition

f = FooBar();
if (f.blah(3) != 3)
  error("FooBar::blah(int)")
endif

if (f.blah(3.5) != 3.5)
  error("FooBar::blah(double)")
endif

if (!strcmp(f.blah("hello"),"hello"))
  error("FooBar::blah(char *)")
endif


f = FooBar2();
if (f.blah(3) != 3)
  error("FooBar2::blah(int)")
endif

if (f.blah(3.5) != 3.5)
  error("FooBar2::blah(double)")
endif

if (!strcmp(f.blah("hello"),"hello"))
  error("FooBar2::blah(char *)")
endif


f = FooBar3();
if (f.blah(3) != 3)
  error("FooBar3::blah(int)")
endif

if (f.blah(3.5) != 3.5)
  error("FooBar3::blah(double)")
endif

if (!strcmp(f.blah("hello"),"hello"))
  error("FooBar3::blah(char *)")
endif

