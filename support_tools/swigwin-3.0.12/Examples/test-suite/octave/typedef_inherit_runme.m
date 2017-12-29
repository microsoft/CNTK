# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

typedef_inherit

a = typedef_inherit.Foo();
b = typedef_inherit.Bar();

x = typedef_inherit.do_blah(a);
if (!strcmp(x,"Foo::blah"))
    error("Whoa! Bad return", x)
endif

x = typedef_inherit.do_blah(b);
if (!strcmp(x,"Bar::blah"))
    error("Whoa! Bad return", x)
endif

c = typedef_inherit.Spam();
d = typedef_inherit.Grok();

x = typedef_inherit.do_blah2(c);
if (!strcmp(x,"Spam::blah"))
    error("Whoa! Bad return", x)
endif

x = typedef_inherit.do_blah2(d);
if (!strcmp(x,"Grok::blah"))
    error("Whoa! Bad return", x)
endif
