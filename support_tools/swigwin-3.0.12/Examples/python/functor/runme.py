# Operator overloading example
import example
import math

a = example.intSum(0)
b = example.doubleSum(100.0)

# Use the objects.  They should be callable just like a normal
# python function.

for i in range(0, 100):
    a(i)                # Note: function call
    b(math.sqrt(i))     # Note: function call

print a.result()
print b.result()
