from li_std_vector_ptr import *

def check(val1, val2):
    if val1 != val2:
        raise RuntimeError("Values are not the same %s %s" % (val1, val2))
ip1 = makeIntPtr(11)
ip2 = makeIntPtr(22)

vi = IntPtrVector((ip1, ip2))
check(getValueFromVector(vi, 0), 11)
check(getValueFromVector(vi, 1), 22)

vA = APtrVector([makeA(33), makeA(34)])
check(getVectorValueA(vA, 0), 33)

vB = BPtrVector([makeB(133), makeB(134)])
check(getVectorValueB(vB, 0), 133)

vC = CPtrVector([makeC(1133), makeC(1134)])
check(getVectorValueC(vC, 0), 1133)


vA = [makeA(233), makeA(234)]
check(getVectorValueA(vA, 0), 233)

vB = [makeB(333), makeB(334)]
check(getVectorValueB(vB, 0), 333)

vC = [makeC(3333), makeC(3334)]
check(getVectorValueC(vC, 0), 3333)

# mixed A and B should not be accepted
vAB = [makeA(999), makeB(999)]
try:
  check(getVectorValueA(vAB, 0), 999)
  raise RuntimeError("missed exception")
except TypeError:
  pass

b111 = makeB(111)
bNones = BPtrVector([None, b111, None])

bCount = 0
noneCount = 0
for b in bNones:
  if b == None:
    noneCount = noneCount + 1
  else:
    if b.val != 111:
      raise RuntimeError("b.val is wrong")
    bCount = bCount + 1

if bCount != 1:
  raise RuntimeError("bCount wrong")
if noneCount != 2:
  raise RuntimeError("noneCount wrong")
