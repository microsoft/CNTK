from implicittest import *


def check(a, b):
    if a != b:
        raise RuntimeError(str(a) + " does not equal " + str(b))


def is_new_style_class(cls):
    return hasattr(cls, "__class__")

#### Class ####

# No implicit conversion
check(1, A(1).get())
check(2, A(1.0).get())
check(3, A(B()).get())
check(4, A("hello").get())
try:
    check(3, A(None).get())
    raise RuntimeError
except ValueError:
    # ValueError: invalid null reference in method 'new_A', argument 1 of type 'B const &'
    # Arguably A(char *) should be chosen, but there is a bug to do with None passed to methods overloaded by value,
    # references and pointers to different types, where pointers ought to be
    # given a slightly higher precedence.
    pass

check(1, get(1))
check(2, get(1.0))
check(3, get(B()))

# Explicit constructor:
try:
    check(4, get("hello"))
    raise RuntimeError
except TypeError:
    pass

#### Template Class ####

# No implicit conversion
check(1, A_int(1).get())
check(2, A_int(1.0).get())
check(3, A_int(B()).get())
check(4, A_int("hello").get())

if is_new_style_class(A_int):
    A_int_static = A_int
else:
    A_int_static = A_int(0)
check(1, A_int_static.sget(1))
check(2, A_int_static.sget(1.0))
check(3, A_int_static.sget(B()))

# explicit constructor:
try:
    check(4, A_int_static.sget("hello"))
    raise RuntimeError
except TypeError:
    pass

#### Global variable assignment ####

cvar.foo = Foo(1)
check(cvar.foo.ii, 1)
cvar.foo = 1
check(cvar.foo.ii, 1)
cvar.foo = 1.0
check(cvar.foo.ii, 2)
cvar.foo = Foo("hello")
check(cvar.foo.ii, 3)

# explicit constructor:
try:
    cvar.foo = "hello"
    raise RuntimeError
except TypeError:
    pass

#### Member variable assignment ####
# Note: also needs naturalvar

b = Bar()
check(b.f.ii, 0)
b.f = Foo("hello")
check(b.f.ii, 3)
b.f = 1
check(b.f.ii, 1)
b.f = 1.0
check(b.f.ii, 2)

# explicit constructor:
try:
    b.f = "hello"
    raise RuntimeError
except TypeError:
    pass

#### Class testing None ####

# No implicit conversion
check(1, AA(1).get())
check(2, AA(1.0).get())
check(3, AA(B()).get())
check(3, AA(None).get())
check(4, AA("hello").get())
check(5, AA(BB()).get())

check(1, get_AA_val(1))
check(2, get_AA_val(1.0))
check(3, get_AA_val(B()))
check(3, get_AA_val(None))
check(5, get_AA_val(BB()))

# Explicit constructor:
try:
    check(4, get_AA_val("hello"))
    raise RuntimeError
except TypeError:
    pass

check(1, get_AA_ref(1))
check(2, get_AA_ref(1.0))
check(3, get_AA_ref(B()))
check(3, get_AA_ref(None))
check(5, get_AA_ref(BB()))

# Explicit constructor:
try:
    check(4, get_AA_ref("hello"))
    raise RuntimeError
except TypeError:
    pass


### overloading priority test ###

ccc = CCC(B())
check(ccc.checkvalue, 10)
check(ccc.xx(123), 11)
check(ccc.yy(123, 123), 111)
