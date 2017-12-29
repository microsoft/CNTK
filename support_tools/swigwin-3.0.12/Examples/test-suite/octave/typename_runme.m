# do not dump Octave core
if exist("crash_dumps_octave_core", "builtin")
  crash_dumps_octave_core(0);
endif

typename
f = typename.Foo();
b = typename.Bar();

x = typename.twoFoo(f);
if (x == floor(x))
    error("Wrong return type!")
endif
y = typename.twoBar(b);
if (y != floor(y))
    error("Wrong return type!")
endif
    
