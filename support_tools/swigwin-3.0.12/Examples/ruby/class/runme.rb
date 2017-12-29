# file: runme.rb

# This file illustrates the C++ interface created by SWIG.
# All of our C++ classes get converted into Ruby classes.

require 'example'

# ----- Object creation -----

print "Creating some objects:\n"
c = Example::Circle.new(10)
print "    Created circle #{c}\n"
s = Example::Square.new(10)
print "    Created square #{s}\n"

# ----- Access a static member -----

print "\nA total of #{Example::Shape.nshapes} shapes were created\n"

# ----- Member data access -----

# Set the location of the object

# Notice how we can do this using functions specific to
# the 'Circle' class.
c.x = 20
c.y = 30

# Now use the same functions in the base class
s.x = -10
s.y = 5

print "\nHere is their current position:\n"
print "    Circle = (", c.x, ",", c.y, ")\n"
print "    Square = (", s.x, ",", s.y, ")\n"

# ----- Call some methods -----

print "\nHere are some properties of the shapes:\n"
for o in [c, s]
  print "    #{o}\n"
  print "        area      = ", o.area, "\n"
  print "        perimeter = ", o.perimeter, "\n"
end
# Notice how the Shape#area() and Shape#perimeter() functions really
# invoke the appropriate virtual method on each object.

# Remove references to the object and force a garbage collection run.
c = s = o = nil
GC.start()

print "\n", Example::Shape.nshapes," shapes remain\n"
print "Goodbye\n"
