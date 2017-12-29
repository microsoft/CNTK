# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

constover

p = constover.test("test");
if (!strcmp(p,"test"))
    error("test failed!")
endif

p = constover.test_pconst("test");
if (!strcmp(p,"test_pconst"))
    error("test_pconst failed!")
endif
    
f = constover.Foo();
p = f.test("test");
if (!strcmp(p,"test"))
    error("member-test failed!")
endif

p = f.test_pconst("test");
if (!strcmp(p,"test_pconst"))
    error("member-test_pconst failed!")
endif

p = f.test_constm("test");
if (!strcmp(p,"test_constmethod"))
    error("member-test_constm failed!")
endif

p = f.test_pconstm("test");
if (!strcmp(p,"test_pconstmethod"))
    error("member-test_pconstm failed!")
endif


