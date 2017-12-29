import python_richcompare

base1 = python_richcompare.BaseClass(1)
base2 = python_richcompare.BaseClass(2)
base3 = python_richcompare.BaseClass(3)
a1 = python_richcompare.SubClassA(1)
a2 = python_richcompare.SubClassA(2)
a3 = python_richcompare.SubClassA(3)
b1 = python_richcompare.SubClassB(1)
b2 = python_richcompare.SubClassB(2)
b3 = python_richcompare.SubClassB(3)

# Check == and != within a single type
#-------------------------------------------------------------------------

if not (base1 == base1):
    raise RuntimeError("Object not == to itself")

if not (base1 == python_richcompare.BaseClass(1)):
    raise RuntimeError("Object not == to an equivalent object")

if (base1 == base2):
    raise RuntimeError(
        "Comparing non-equivalent objects of the same type, == returned True")

if (base1 != base1):
    raise RuntimeError("Object is != itself")

if (base1 != python_richcompare.BaseClass(1)):
    raise RuntimeError("Object is != an equivalent object")

if not (base1 != base2):
    raise RuntimeError(
        "Comparing non-equivalent objects of the same type, != returned False")


# Check redefined operator== in SubClassA
#-------------------------------------------------------------------------

if (a2 == base2):
    raise RuntimeError("Redefined operator== in SubClassA failed")

if (a2 == b2):
    raise RuntimeError("Redefined operator== in SubClassA failed")

if not (a1 == a2):
    raise RuntimeError("Redefined operator== in SubClassA failed")

# Check up-casting of subclasses
#-------------------------------------------------------------------------

if (base2 != a2):
    raise RuntimeError(
        "Comparing equivalent base and subclass instances, != returned True")

if (a2 == base2):
    raise RuntimeError(
        "Comparing non-equivalent base and subclass instances, == returned True")

if (a1 == b1):
    raise RuntimeError(
        "Comparing equivalent instances of different subclasses, == returned True")

if (b1 == a1):
    raise RuntimeError(
        "Comparing equivalent instances of different subclasses, == returned True")

# Check inequalities
#-------------------------------------------------------------------------

if (a2 > b2):
    raise RuntimeError("operator> failed")

if (a2 < b2):
    raise RuntimeError("operator< failed")

if not (a2 >= b2):
    raise RuntimeError("operator>= failed")

if not (a2 <= b2):
    raise RuntimeError("operator<= failed")

# Check inequalities used for ordering
#-------------------------------------------------------------------------

x = sorted([a2, a3, a1])

if not (x[0] is a1):
    raise RuntimeError("Ordering failed")

if not (x[1] is a2):
    raise RuntimeError("Ordering failed")

if not (x[2] is a3):
    raise RuntimeError("Ordering failed")

x = sorted([base2, a3, b1])

if not (x[0] is b1):
    raise RuntimeError("Ordering failed")

if not (x[1] is base2):
    raise RuntimeError("Ordering failed")

if not (x[2] is a3):
    raise RuntimeError("Ordering failed")
