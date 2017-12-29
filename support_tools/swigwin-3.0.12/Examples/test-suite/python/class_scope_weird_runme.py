import class_scope_weird

f = class_scope_weird.Foo()
g = class_scope_weird.Foo(3)
if f.bar(3) != 3:
    raise RuntimeError
