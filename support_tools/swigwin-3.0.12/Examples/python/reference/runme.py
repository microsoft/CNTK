# file: runme.py

# This file illustrates the manipulation of C++ references in Python

import example

# ----- Object creation -----

print "Creating some objects:"
a = example.Vector(3, 4, 5)
b = example.Vector(10, 11, 12)

print "    Created", a.cprint()
print "    Created", b.cprint()

# ----- Call an overloaded operator -----

# This calls the wrapper we placed around
#
#      operator+(const Vector &a, const Vector &)
#
# It returns a new allocated object.

print "Adding a+b"
c = example.addv(a, b)
print "    a+b =", c.cprint()

# Note: Unless we free the result, a memory leak will occur
del c

# ----- Create a vector array -----

# Note: Using the high-level interface here
print "Creating an array of vectors"
va = example.VectorArray(10)
print "    va = ", va

# ----- Set some values in the array -----

# These operators copy the value of $a and $b to the vector array
va.set(0, a)
va.set(1, b)

va.set(2, example.addv(a, b))

# Get some values from the array

print "Getting some array values"
for i in range(0, 5):
    print "    va(%d) = %s" % (i, va.get(i).cprint())

# Watch under resource meter to check on this
print "Making sure we don't leak memory."
for i in xrange(0, 1000000):
    c = va.get(i % 10)

# ----- Clean up -----
print "Cleaning up"

del va
del a
del b
