import extend_template

f = extend_template.Foo_0()
if f.test1(37) != 37:
    raise RuntimeError

if f.test2(42) != 42:
    raise RuntimeError
