from smart_pointer_const_overload import *

CONST_ACCESS = 1
MUTABLE_ACCESS = 2


def test(b, f):
    if f.x != 0:
        raise RuntimeError

    # Test member variable get
    if b.x != 0:
        raise RuntimeError

    if f.access != CONST_ACCESS:
        raise RuntimeError

    # Test member variable set
    b.x = 1

    if f.x != 1:
        raise RuntimeError

    if f.access != MUTABLE_ACCESS:
        raise RuntimeError

    # Test const method
    if b.getx() != 1:
        raise RuntimeError

    if f.access != CONST_ACCESS:
        raise RuntimeError

    # Test mutable method
    b.setx(2)

    if f.x != 2:
        raise RuntimeError

    if f.access != MUTABLE_ACCESS:
        raise RuntimeError

    # Test extended const method
    if b.getx2() != 2:
        raise RuntimeError

    if f.access != CONST_ACCESS:
        raise RuntimeError

    # Test extended mutable method
    b.setx2(3)

    if f.x != 3:
        raise RuntimeError

    if f.access != MUTABLE_ACCESS:
        raise RuntimeError

    # Test static method
    b.statMethod()

    if f.access != CONST_ACCESS:
        raise RuntimeError

    # Test const member
    f.access = MUTABLE_ACCESS

    if b.y != 0:
        raise RuntimeError

    if f.access != CONST_ACCESS:
        raise RuntimeError

    # Test get through mutable pointer to const member
    f.access = MUTABLE_ACCESS

    if get_int(b.yp) != 0:
        raise RuntimeError

    if f.access != CONST_ACCESS:
        raise RuntimeError

    # Test get through const pointer to mutable member
    f.x = 4
    f.access = MUTABLE_ACCESS

    if get_int(b.xp) != 4:
        raise RuntimeError

    if f.access != CONST_ACCESS:
        raise RuntimeError

    # Test set through const pointer to mutable member
    f.access = MUTABLE_ACCESS
    set_int(b.xp, 5)

    if f.x != 5:
        raise RuntimeError

    if f.access != CONST_ACCESS:
        raise RuntimeError

    # Test set pointer to const member
    b.yp = new_int(6)

    if f.y != 0:
        raise RuntimeError

    if get_int(f.yp) != 6:
        raise RuntimeError

    if f.access != MUTABLE_ACCESS:
        raise RuntimeError

    delete_int(f.yp)

f = Foo()
b = Bar(f)

f2 = Foo()
b2 = Bar2(f2)

test(b, f)
test(b2, f2)
