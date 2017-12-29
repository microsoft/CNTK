import typemap_documentation

f = typemap_documentation.Foo()
f.x = 55
b = typemap_documentation.Bar()
b.y = 44

if 55 != typemap_documentation.GrabVal(f):
    raise RuntimeError("bad value")

try:
    typemap_documentation.GrabVal(b)
    raise RuntimeError("unexpected exception")
except TypeError:
    pass

if 55 != typemap_documentation.GrabValFooBar(f):
    raise RuntimeError("bad f value")
if 44 != typemap_documentation.GrabValFooBar(b):
    raise RuntimeError("bad b value")
