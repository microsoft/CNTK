# file: runme.py

import example

# Call our gcd() function

x = 42
y = 105
g = example.gcd(x, y)
print "The gcd of %d and %d is %d" % (x, y, g)

# Manipulate the Foo global variable

# Output its current value
print "Foo = ", example.cvar.Foo

# Change its value
example.cvar.Foo = 3.1415926

# See if the change took effect
print "Foo = ", example.cvar.Foo
