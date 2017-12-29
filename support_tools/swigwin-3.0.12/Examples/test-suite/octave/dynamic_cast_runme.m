# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

dynamic_cast

f = dynamic_cast.Foo();
b = dynamic_cast.Bar();

x = f.blah();
y = b.blah();

a = dynamic_cast.do_test(y);
if (!strcmp(a,"Bar::test"))
  error("Failed!!")
endif

    
