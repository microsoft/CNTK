import cpp_enum

f = cpp_enum.Foo()

if f.hola != f.Hello:
    print f.hola
    raise RuntimeError

f.hola = f.Hi
if f.hola != f.Hi:
    print f.hola
    raise RuntimeError

f.hola = f.Hello

if f.hola != f.Hello:
    print f.hola
    raise RuntimeError

cpp_enum.cvar.hi = cpp_enum.Hello
if cpp_enum.cvar.hi != cpp_enum.Hello:
    print cpp_enum.cvar.hi
    raise RuntimeError
