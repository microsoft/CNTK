# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

using_extend

f = FooBar();
if (f.blah(3) != 3)
    error("blah(int)")
endif

if (f.blah(3.5) != 3.5)
    error("blah(double)")
endif

if (!strcmp(f.blah("hello"),"hello"))
    error("blah(char *)")
endif

if (f.blah(3,4) != 7)
    error("blah(int,int)")
endif

if (f.blah(3.5,7.5) != (3.5+7.5))
    error("blah(double,double)")
endif


if (f.duh(3) != 3)
    error("duh(int)")
endif
