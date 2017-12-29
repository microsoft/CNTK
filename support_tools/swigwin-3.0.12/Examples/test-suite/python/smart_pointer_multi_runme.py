from smart_pointer_multi import *

f = Foo()
b = Bar(f)
s = Spam(b)
g = Grok(b)

s.x = 3
if s.getx() != 3:
    raise RuntimeError

g.x = 4
if g.getx() != 4:
    raise RuntimeError
