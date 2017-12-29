import constover
import sys
error = 0

p = constover.test("test")
if p != "test":
    print "test failed!"
    error = 1

p = constover.test_pconst("test")
if p != "test_pconst":
    print "test_pconst failed!"
    error = 1

f = constover.Foo()
p = f.test("test")
if p != "test":
    print "member-test failed!"
    error = 1

p = f.test_pconst("test")
if p != "test_pconst":
    print "member-test_pconst failed!"
    error = 1

p = f.test_constm("test")
if p != "test_constmethod":
    print "member-test_constm failed!"
    error = 1

p = f.test_pconstm("test")
if p != "test_pconstmethod":
    print "member-test_pconstm failed!"
    error = 1

sys.exit(error)
