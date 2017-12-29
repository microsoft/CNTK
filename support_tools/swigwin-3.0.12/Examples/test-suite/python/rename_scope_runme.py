from rename_scope import *

a = Natural_UP()
b = Natural_BP()

if a.rtest() != 1:
    raise RuntimeError

if b.rtest() != 1:
    raise RuntimeError

f = equals
