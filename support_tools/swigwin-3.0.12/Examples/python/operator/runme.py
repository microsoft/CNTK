# Operator overloading example
import example

a = example.Complex(2, 3)
b = example.Complex(-5, 10)

print "a   =", a
print "b   =", b

c = a + b
print "c   =", c
print "a*b =", a * b
print "a-c =", a - c

e = example.ComplexCopy(a - c)
print "e   =", e

# Big expression
f = ((a + b) * (c + b * e)) + (-a)
print "f   =", f
