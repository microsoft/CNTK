import complextest

a = complex(-1, 2)

if complextest.Conj(a) != a.conjugate():
    raise RuntimeError, "bad complex mapping"

if complextest.Conjf(a) != a.conjugate():
    raise RuntimeError, "bad complex mapping"


v = (complex(1, 2), complex(2, 3), complex(4, 3), 1)

try:
    complextest.Copy_h(v)
except:
    pass
