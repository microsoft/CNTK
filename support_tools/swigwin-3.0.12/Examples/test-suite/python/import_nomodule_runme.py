from import_nomodule import *

# This test is expected to fail with -builtin option.
# The base class is needed for the builtin class hierarchy
if is_python_builtin():
    exit(0)

f = create_Foo()
test1(f, 42)
delete_Foo(f)

b = Bar()
test1(b, 37)
