cpp_enum

f = cpp_enum.Foo();

if (f.hola != cpp_enum.Foo_Hello)
     error(f.hola);
     error;
endif

f.hola = cpp_enum.Foo_Hi;
if (f.hola != cpp_enum.Foo_Hi)
     error(f.hola);
     error;
endif

f.hola = cpp_enum.Foo_Hello;

if (f.hola != cpp_enum.Foo_Hello)
     error(f.hola);
     error;
endif

cpp_enum.hi = cpp_enum.Hello;
if (cpp_enum.hi != cpp_enum.Hello)
     error(cpp_enum.hi);
     error;
endif
