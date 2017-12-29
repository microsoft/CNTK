import contract

contract.test_preassert(1, 2)
try:
    contract.test_preassert(-1)
    print "Failed! Preassertions are broken"
except:
    pass

contract.test_postassert(3)
try:
    contract.test_postassert(-3)
    print "Failed! Postassertions are broken"
except:
    pass

contract.test_prepost(2, 3)
contract.test_prepost(5, -4)
try:
    contract.test_prepost(-3, 4)
    print "Failed! Preassertions are broken"
except:
    pass

try:
    contract.test_prepost(4, -10)
    print "Failed! Postassertions are broken"

except:
    pass

f = contract.Foo()
f.test_preassert(4, 5)
try:
    f.test_preassert(-2, 3)
    print "Failed! Method preassertion."
except:
    pass

f.test_postassert(4)
try:
    f.test_postassert(-4)
    print "Failed! Method postassertion"
except:
    pass

f.test_prepost(3, 4)
f.test_prepost(4, -3)
try:
    f.test_prepost(-4, 2)
    print "Failed! Method preassertion."
except:
    pass

try:
    f.test_prepost(4, -10)
    print "Failed! Method postassertion."
except:
    pass

contract.Foo_stest_prepost(4, 0)
try:
    contract.Foo_stest_prepost(-4, 2)
    print "Failed! Static method preassertion"
except:
    pass

try:
    contract.Foo_stest_prepost(4, -10)
    print "Failed! Static method posteassertion"
except:
    pass

b = contract.Bar()
try:
    b.test_prepost(2, -4)
    print "Failed! Inherited preassertion."
except:
    pass


d = contract.D()
try:
    d.foo(-1, 1, 1, 1, 1)
    print "Failed! Inherited preassertion (D)."
except:
    pass
try:
    d.foo(1, -1, 1, 1, 1)
    print "Failed! Inherited preassertion (D)."
except:
    pass
try:
    d.foo(1, 1, -1, 1, 1)
    print "Failed! Inherited preassertion (D)."
except:
    pass
try:
    d.foo(1, 1, 1, -1, 1)
    print "Failed! Inherited preassertion (D)."
except:
    pass
try:
    d.foo(1, 1, 1, 1, -1)
    print "Failed! Inherited preassertion (D)."
except:
    pass


try:
    d.bar(-1, 1, 1, 1, 1)
    print "Failed! Inherited preassertion (D)."
except:
    pass
try:
    d.bar(1, -1, 1, 1, 1)
    print "Failed! Inherited preassertion (D)."
except:
    pass
try:
    d.bar(1, 1, -1, 1, 1)
    print "Failed! Inherited preassertion (D)."
except:
    pass
try:
    d.bar(1, 1, 1, -1, 1)
    print "Failed! Inherited preassertion (D)."
except:
    pass
try:
    d.bar(1, 1, 1, 1, -1)
    print "Failed! Inherited preassertion (D)."
except:
    pass

# Namespace
my = contract.myClass(1)
try:
    my = contract.myClass(0)
    print "Failed! constructor preassertion"
except:
    pass
