from refcount import *
#
# very innocent example
#

a = A3()
b1 = B(a)
b2 = B_create(a)


if a.ref_count() != 3:
    raise RuntimeError("Count = %d" % a.ref_count())


rca = b2.get_rca()
b3 = B_create(rca)

if a.ref_count() != 5:
    raise RuntimeError("Count = %d" % a.ref_count())


v = vector_A(2)
v[0] = a
v[1] = a

x = v[0]
del v

if a.ref_count() != 6:
    raise RuntimeError("Count = %d" % a.ref_count())

# Check %newobject
b4 = b2.cloner()
if b4.ref_count() != 1:
    raise RuntimeError

b5 = global_create(a)
if b5.ref_count() != 1:
    raise RuntimeError

b6 = Factory_create(a)
if b6.ref_count() != 1:
    raise RuntimeError

b7 = Factory().create2(a)
if b7.ref_count() != 1:
    raise RuntimeError


if a.ref_count() != 10:
    raise RuntimeError("Count = %d" % a.ref_count())

del b4
del b5
del b6
del b7

if a.ref_count() != 6:
    raise RuntimeError("Count = %d" % a.ref_count())
