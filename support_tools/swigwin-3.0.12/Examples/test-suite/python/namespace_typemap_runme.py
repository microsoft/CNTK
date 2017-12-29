from namespace_typemap import *

if stest1("hello") != "hello":
    raise RuntimeError

if stest2("hello") != "hello":
    raise RuntimeError

if stest3("hello") != "hello":
    raise RuntimeError

if stest4("hello") != "hello":
    raise RuntimeError

if stest5("hello") != "hello":
    raise RuntimeError

if stest6("hello") != "hello":
    raise RuntimeError

if stest7("hello") != "hello":
    raise RuntimeError

if stest8("hello") != "hello":
    raise RuntimeError

if stest9("hello") != "hello":
    raise RuntimeError

if stest10("hello") != "hello":
    raise RuntimeError

if stest11("hello") != "hello":
    raise RuntimeError

if stest12("hello") != "hello":
    raise RuntimeError

c = complex(2, 3)
r = c.real

if ctest1(c) != r:
    raise RuntimeError

if ctest2(c) != r:
    raise RuntimeError

if ctest3(c) != r:
    raise RuntimeError

if ctest4(c) != r:
    raise RuntimeError

if ctest5(c) != r:
    raise RuntimeError

if ctest6(c) != r:
    raise RuntimeError

if ctest7(c) != r:
    raise RuntimeError

if ctest8(c) != r:
    raise RuntimeError

if ctest9(c) != r:
    raise RuntimeError

if ctest10(c) != r:
    raise RuntimeError

if ctest11(c) != r:
    raise RuntimeError

if ctest12(c) != r:
    raise RuntimeError

try:
    ttest1(-14)
    raise RuntimeError
except ValueError:
    pass
