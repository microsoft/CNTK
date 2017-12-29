# file: runme.rb

require 'example'

# Call our gcd() function

x = 42
y = 105
g = Example.gcd(x,y)
printf "The gcd of %d and %d is %d\n",x,y,g

# Manipulate the Foo global variable

# Output its current value
print "Foo = ", Example.Foo, "\n"

# Change its value
Example.Foo = 3.1415926

# See if the change took effect
print "Foo = ", Example.Foo, "\n"
