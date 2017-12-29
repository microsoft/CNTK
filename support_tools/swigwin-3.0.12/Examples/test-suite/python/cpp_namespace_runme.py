# Note: This example assumes that namespaces are flattened
import cpp_namespace

n = cpp_namespace.fact(4)
if n != 24:
    raise RuntimeError("Bad return value!")

if cpp_namespace.cvar.Foo != 42:
    raise RuntimeError("Bad variable value!")

t = cpp_namespace.Test()
if t.method() != "Test::method":
    raise RuntimeError("Bad method return value!")

if cpp_namespace.do_method(t) != "Test::method":
    raise RuntimeError("Bad return value!")

if cpp_namespace.do_method2(t) != "Test::method":
    raise RuntimeError("Bad return value!")

cpp_namespace.weird("hello", 4)

del t

t2 = cpp_namespace.Test2()
t3 = cpp_namespace.Test3()
t4 = cpp_namespace.Test4()
t5 = cpp_namespace.Test5()

if cpp_namespace.foo3(42) != 42:
    raise RuntimeError("Bad return value!")

if cpp_namespace.do_method3(t2, 40) != "Test2::method":
    raise RuntimeError("Bad return value!")

if cpp_namespace.do_method3(t3, 40) != "Test3::method":
    raise RuntimeError("Bad return value!")

if cpp_namespace.do_method3(t4, 40) != "Test4::method":
    raise RuntimeError("Bad return value!")

if cpp_namespace.do_method3(t5, 40) != "Test5::method":
    raise RuntimeError("Bad return value!")
