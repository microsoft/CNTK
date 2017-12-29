import _profiletest
#import profiletest

pa = _profiletest.new_A()
pb = _profiletest.new_B()
fn = _profiletest.B_fn
destroy = _profiletest.delete_A
i = 50000
a = pa
while i:
    a = fn(pb, a)  # 1
    destroy(a)
    a = fn(pb, a)  # 2
    destroy(a)
    a = fn(pb, a)  # 3
    destroy(a)
    a = fn(pb, a)  # 4
    destroy(a)
    a = fn(pb, a)  # 5
    destroy(a)
    a = fn(pb, a)  # 6
    destroy(a)
    a = fn(pb, a)  # 7
    destroy(a)
    a = fn(pb, a)  # 8
    destroy(a)
    a = fn(pb, a)  # 9
    destroy(a)
    a = fn(pb, a)  # 10
    destroy(a)
    a = fn(pb, a)  # 1
    destroy(a)
    a = fn(pb, a)  # 2
    destroy(a)
    a = fn(pb, a)  # 3
    destroy(a)
    a = fn(pb, a)  # 4
    destroy(a)
    a = fn(pb, a)  # 5
    destroy(a)
    a = fn(pb, a)  # 6
    destroy(a)
    a = fn(pb, a)  # 7
    destroy(a)
    a = fn(pb, a)  # 8
    destroy(a)
    a = fn(pb, a)  # 9
    destroy(a)
    a = fn(pb, a)  # 20
    destroy(a)
    i -= 1

_profiletest.delete_A(pa)
_profiletest.delete_B(pb)
