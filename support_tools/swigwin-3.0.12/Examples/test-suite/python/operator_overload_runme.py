from operator_overload import *

# first check all the operators are implemented correctly from pure C++ code
Op_sanity_check()

pop = Op(6)/Op(3)

# test routine:
a=Op()
b=Op(5)
c=Op(b) # copy construct
d=Op(2)
dd=d # assignment operator

# test equality
if not a!=b:
  raise RuntimeError("a!=b")
if not b==c:
  raise RuntimeError("b==c")
if not a!=d:
  raise RuntimeError("a!=d")
if not d==dd:
  raise RuntimeError("d==dd")

# test <
if not a<b:
  raise RuntimeError("a<b")
if not a<=b:
  raise RuntimeError("a<=b")
if not b<=c:
  raise RuntimeError("b<=c")
if not b>=c:
  raise RuntimeError("b>=c")
if not b>d:
  raise RuntimeError("b>d")
if not b>=d:
  raise RuntimeError("b>=d")

# test +=
e=Op(3)
e+=d
if not e==b:
  raise RuntimeError("e==b (%s==%s)" % (e.i, b.i))
e-=c
if not e==a:
  raise RuntimeError("e==a")
e=Op(1)
e*=b
if not e==c:
  raise RuntimeError("e==c")
e/=d
if not e==d:
  raise RuntimeError("e==d")
e%=c;
if not e==d:
  raise RuntimeError("e==d")

# test +
f=Op(1)
g=Op(1)
if not f+g==Op(2):
  raise RuntimeError("f+g==Op(2)")
if not f-g==Op(0):
  raise RuntimeError("f-g==Op(0)")
if not f*g==Op(1):
  raise RuntimeError("f*g==Op(1)")
if not f/g==Op(1):
  raise RuntimeError("f/g==Op(1)")
if not f%g==Op(0):
  raise RuntimeError("f%g==Op(0)")

# test unary operators
if not -a==a:
  raise RuntimeError("-a==a")
if not -b==Op(-5):
  raise RuntimeError("-b==Op(-5)")

