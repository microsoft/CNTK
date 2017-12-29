from li_std_vector_extra import *

iv = IntVector(4)
for i in range(0, 4):
    iv[i] = i

x = average(iv)
y = average([1, 2, 3, 4])

a = half([10, 10.5, 11, 11.5])

dv = DoubleVector(10)
for i in range(0, 10):
    dv[i] = i / 2.0

halve_in_place(dv)


bv = BoolVector(4)
bv[0] = bool(1)
bv[1] = bool(0)
bv[2] = bool(4)
bv[3] = bool(0)

if bv[0] != bv[2]:
    raise RuntimeError, "bad std::vector<bool> mapping"

b = B(5)
va = VecA([b, None, b, b])

if va[0].f(1) != 6:
    raise RuntimeError, "bad std::vector<A*> mapping"

if vecAptr(va) != 6:
    raise RuntimeError, "bad std::vector<A*> mapping"

b.val = 7
if va[3].f(1) != 8:
    raise RuntimeError, "bad std::vector<A*> mapping"


ip = PtrInt()
ap = new_ArrInt(10)

ArrInt_setitem(ip, 0, 123)
ArrInt_setitem(ap, 2, 123)

vi = IntPtrVector((ip, ap, None))
if ArrInt_getitem(vi[0], 0) != ArrInt_getitem(vi[1], 2):
    raise RuntimeError, "bad std::vector<int*> mapping"

delete_ArrInt(ap)


a = halfs([10, 8, 4, 3])

v = IntVector()
v[0:2] = [1, 2]
if v[0] != 1 or v[1] != 2:
    raise RuntimeError, "bad setslice"

if v[0:-1][0] != 1:
    raise RuntimeError, "bad getslice"

if v[0:-2].size() != 0:
    raise RuntimeError, "bad getslice"

v[0:1] = [2]
if v[0] != 2:
    raise RuntimeError, "bad setslice"

v[1:] = [3]
if v[1] != 3:
    raise RuntimeError, "bad setslice"

v[2:] = [3]
if v[2] != 3:
    raise RuntimeError, "bad setslice"

if v[0:][0] != v[0]:
    raise RuntimeError, "bad getslice"


del v[:]
if v.size() != 0:
    raise RuntimeError, "bad getslice"

del v[:]
if v.size() != 0:
    raise RuntimeError, "bad getslice"


v = vecStr(["hello ", "world"])
if v[0] != 'hello world':
    raise RuntimeError, "bad std::string+std::vector"


pv = pyvector([1, "hello", (1, 2)])

if pv[1] != "hello":
    raise RuntimeError


iv = IntVector(5)
for i in range(0, 5):
    iv[i] = i

iv[1:3] = []
if iv[1] != 3:
    raise RuntimeError

# Overloading checks
if overloaded1(iv) != "vector<int>":
    raise RuntimeError

if overloaded1(dv) != "vector<double>":
    raise RuntimeError

if overloaded2(iv) != "vector<int>":
    raise RuntimeError

if overloaded2(dv) != "vector<double>":
    raise RuntimeError

if overloaded3(iv) != "vector<int> *":
    raise RuntimeError

if overloaded3(None) != "vector<int> *":
    raise RuntimeError

if overloaded3(100) != "int":
    raise RuntimeError


# vector pointer checks
ip = makeIntPtr(11)
dp = makeDoublePtr(33.3)
error = 0
try:
    # check vector<int *> does not accept double * element
    vi = IntPtrVector((ip, dp))
    error = 1
except:
    pass

if error:
    raise RuntimeError

vi = IntPtrVector((ip, makeIntPtr(22)))
if extractInt(vi[0]) != 11:
    raise RuntimeError

if extractInt(vi[1]) != 22:
    raise RuntimeError

# vector const pointer checks
csp = makeConstShortPtr(111)

error = 0
try:
    # check vector<const unsigned short *> does not accept double * element
    vcs = ConstShortPtrVector((csp, dp))
    error = 1
except:
    pass

if error:
    raise RuntimeError

vcs = ConstShortPtrVector((csp, makeConstShortPtr(222)))
if extractConstShort(vcs[0]) != 111:
    raise RuntimeError

if extractConstShort(vcs[1]) != 222:
    raise RuntimeError
