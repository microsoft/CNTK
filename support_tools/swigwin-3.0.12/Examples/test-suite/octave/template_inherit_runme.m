# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

template_inherit
a = FooInt();
b = FooDouble();
c = BarInt();
d = BarDouble();
e = FooUInt();
f = BarUInt();

if (!strcmp(a.blah(),"Foo"))
    error
endif

if (!strcmp(b.blah(),"Foo"))
    error
endif

if (!strcmp(e.blah(),"Foo"))
    error
endif

if (!strcmp(c.blah(),"Bar"))
    error
endif

if (!strcmp(d.blah(),"Bar"))
    error
endif

if (!strcmp(f.blah(),"Bar"))
    error
endif

if (!strcmp(c.foomethod(),"foomethod"))
    error
endif

if (!strcmp(d.foomethod(),"foomethod"))
    error
endif

if (!strcmp(f.foomethod(),"foomethod"))
    error
endif

if (!strcmp(invoke_blah_int(a),"Foo"))
    error
endif

if (!strcmp(invoke_blah_int(c),"Bar"))
    error
endif

if (!strcmp(invoke_blah_double(b),"Foo"))
    error
endif

if (!strcmp(invoke_blah_double(d),"Bar"))
    error
endif

if (!strcmp(invoke_blah_uint(e),"Foo"))
    error
endif

if (!strcmp(invoke_blah_uint(f),"Bar"))
    error
endif

