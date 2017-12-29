import os.path

testname = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
print "Testing " + testname + " - %module(package=...) + python 'import' in __init__.py"

import pkg1.pkg2.foo
print "  Finished importing pkg1.pkg2.foo"

var2 = pkg1.pkg2.foo.Pkg2_Foo()

classname = str(type(var2))
# Check for an old-style class if swig was run in -classic mode
if classname == "<type 'instance'>":
    classname = str(var2.__class__)

if classname.find("pkg1.pkg2.foo.Pkg2_Foo") == -1:
    raise RuntimeError("failed type checking: " + classname)
print "  Successfully created object pkg1.pkg2.foo.Pkg2_Foo"
