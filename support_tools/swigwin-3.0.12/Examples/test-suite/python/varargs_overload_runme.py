import varargs_overload

if varargs_overload.vararg_over1("Hello") != "Hello":
    raise RuntimeError, "Failed"

if varargs_overload.vararg_over1(2) != "2":
    raise RuntimeError, "Failed"


if varargs_overload.vararg_over2("Hello") != "Hello":
    raise RuntimeError, "Failed"

if varargs_overload.vararg_over2(2, 2.2) != "2 2.2":
    raise RuntimeError, "Failed"


if varargs_overload.vararg_over3("Hello") != "Hello":
    raise RuntimeError, "Failed"

if varargs_overload.vararg_over3(2, 2.2, "hey") != "2 2.2 hey":
    raise RuntimeError, "Failed"

if varargs_overload.vararg_over4("Hello") != "Hello":
    raise RuntimeError, "Failed"

if varargs_overload.vararg_over4(123) != "123":
    raise RuntimeError, "Failed"

if varargs_overload.vararg_over4("Hello", 123) != "Hello":
    raise RuntimeError, "Failed"
