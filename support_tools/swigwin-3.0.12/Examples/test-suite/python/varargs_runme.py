import varargs

if varargs.test("Hello") != "Hello":
    raise RuntimeError, "Failed"

f = varargs.Foo("Greetings")
if f.str != "Greetings":
    raise RuntimeError, "Failed"

if f.test("Hello") != "Hello":
    raise RuntimeError, "Failed"


if varargs.test_def("Hello", 1) != "Hello":
    raise RuntimeError, "Failed"

if varargs.test_def("Hello") != "Hello":
    raise RuntimeError, "Failed"

###
if varargs.test_plenty("Hello") != "Hello":
    raise RuntimeError, "Failed"

if varargs.test_plenty("Hello", 1) != "Hello":
    raise RuntimeError, "Failed"

if varargs.test_plenty("Hello", 1, 2) != "Hello":
    raise RuntimeError, "Failed"

try:
    varargs.test_plenty("Hello", 1, 2, 3)
    raise RuntimeError
except NotImplementedError:
    pass
except TypeError:
    pass
