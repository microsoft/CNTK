import iadd

f = iadd.Foo()

f.AsA.x = 3
f.AsA += f.AsA

if f.AsA.x != 6:
    raise RuntimeError
