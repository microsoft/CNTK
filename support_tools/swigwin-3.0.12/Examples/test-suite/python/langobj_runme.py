import sys
from langobj import *


x = "hello"
rx = sys.getrefcount(x)
v = identity(x)
rv = sys.getrefcount(v)
if v != x:
    raise RuntimeError

if rv - rx != 1:
    raise RuntimeError
