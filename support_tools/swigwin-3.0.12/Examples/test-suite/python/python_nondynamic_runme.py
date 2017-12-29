import python_nondynamic

aa = python_nondynamic.A()

aa.a = 1
aa.b = 2
try:
    aa.c = 2
    err = 0
except:
    err = 1

if not err:
    raise RuntimeError, "A is not static"


class B(python_nondynamic.A):
    c = 4

    def __init__(self):
        python_nondynamic.A.__init__(self)
        pass
    pass

bb = B()

try:
    bb.c = 3
    err = 0
except:
    err = 1

if not err:
    print "bb.c = %d" % bb.c
    print "B.c = %d" % B.c
    raise RuntimeError, "B.c class variable messes up nondynamic-ness of B"

try:
    bb.d = 2
    err = 0
except:
    err = 1

if not err:
    raise RuntimeError, "B is not static"

cc = python_nondynamic.C()
cc.d = 3
