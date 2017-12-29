from overload_simple import *

if foo(3) != "foo:int":
    raise RuntimeError, "foo(int)"

if foo(3.0) != "foo:double":
    raise RuntimeError, "foo(double)"

if foo("hello") != "foo:char *":
    raise RuntimeError, "foo(char *)"

f = Foo()
b = Bar()

if foo(f) != "foo:Foo *":
    raise RuntimeError, "foo(Foo *)"

if foo(b) != "foo:Bar *":
    raise RuntimeError, "foo(Bar *)"

v = malloc_void(32)

if foo(v) != "foo:void *":
    raise RuntimeError, "foo(void *)"

s = Spam()

if s.foo(3) != "foo:int":
    raise RuntimeError, "Spam::foo(int)"

if s.foo(3.0) != "foo:double":
    raise RuntimeError, "Spam::foo(double)"

if s.foo("hello") != "foo:char *":
    raise RuntimeError, "Spam::foo(char *)"

if s.foo(f) != "foo:Foo *":
    raise RuntimeError, "Spam::foo(Foo *)"

if s.foo(b) != "foo:Bar *":
    raise RuntimeError, "Spam::foo(Bar *)"

if s.foo(v) != "foo:void *":
    raise RuntimeError, "Spam::foo(void *)"

if Spam_bar(3) != "bar:int":
    raise RuntimeError, "Spam::bar(int)"

if Spam_bar(3.0) != "bar:double":
    raise RuntimeError, "Spam::bar(double)"

if Spam_bar("hello") != "bar:char *":
    raise RuntimeError, "Spam::bar(char *)"

if Spam_bar(f) != "bar:Foo *":
    raise RuntimeError, "Spam::bar(Foo *)"

if Spam_bar(b) != "bar:Bar *":
    raise RuntimeError, "Spam::bar(Bar *)"

if Spam_bar(v) != "bar:void *":
    raise RuntimeError, "Spam::bar(void *)"

# Test constructors

s = Spam()
if s.type != "none":
    raise RuntimeError, "Spam()"

s = Spam(3)
if s.type != "int":
    raise RuntimeError, "Spam(int)"

s = Spam(3.4)
if s.type != "double":
    raise RuntimeError, "Spam(double)"

s = Spam("hello")
if s.type != "char *":
    raise RuntimeError, "Spam(char *)"

s = Spam(f)
if s.type != "Foo *":
    raise RuntimeError, "Spam(Foo *)"

s = Spam(b)
if s.type != "Bar *":
    raise RuntimeError, "Spam(Bar *)"

s = Spam(v)
if s.type != "void *":
    raise RuntimeError, "Spam(void *)"


free_void(v)


a = ClassA()
b = a.method1(1)
