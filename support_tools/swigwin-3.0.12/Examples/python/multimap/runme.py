# file: runme.py

import example

# Call our gcd() function

x = 42
y = 105
g = example.gcd(x, y)
print "The gcd of %d and %d is %d" % (x, y, g)

# Call the gcdmain() function
example.gcdmain(["gcdmain", "42", "105"])

# Call the count function
print example.count("Hello World", "l")

# Call the capitalize function

print example.capitalize("hello world")
