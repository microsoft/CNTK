from li_implicit import *
b = B()
ai = A(1)
ad = A(2.0)
ab = A(b)

ai, get(ai)
ad, get(ad)
ab, get(ab)

if get(ai) != get(1):
    raise RuntimeError, "bad implicit type"
if get(ad) != get(2.0):
    raise RuntimeError, "bad implicit type"
if get(ab) != get(b):
    raise RuntimeError, "bad implicit type"
