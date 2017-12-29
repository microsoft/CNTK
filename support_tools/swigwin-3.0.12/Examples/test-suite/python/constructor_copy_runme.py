from constructor_copy import *

f1 = Foo1(3)
f11 = Foo1(f1)


if f1.x != f11.x:
    raise RuntimeError


f8 = Foo8()
try:
    f81 = Foo8(f8)
    good = 0
except:
    good = 1

if not good:
    raise RuntimeError


bi = Bari(5)
bc = Bari(bi)

if (bi.x != bc.x):
    raise RuntimeError


bd = Bard(5)
try:
    bc = Bard(bd)
    good = 0
except:
    good = 1

if not good:
    raise RuntimeError
