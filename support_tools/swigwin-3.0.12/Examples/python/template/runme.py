# file: runme.py

import example

# Call some templated functions
print example.maxint(3, 7)
print example.maxdouble(3.14, 2.18)

# Create some class

iv = example.vecint(100)
dv = example.vecdouble(1000)

for i in range(0, 100):
    iv.setitem(i, 2 * i)

for i in range(0, 1000):
    dv.setitem(i, 1.0 / (i + 1))

sum = 0
for i in range(0, 100):
    sum = sum + iv.getitem(i)

print sum

sum = 0.0
for i in range(0, 1000):
    sum = sum + dv.getitem(i)
print sum

del iv
del dv
