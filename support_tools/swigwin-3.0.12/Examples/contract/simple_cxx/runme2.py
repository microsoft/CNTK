import example 

# Create the Circle object

r = 2;
print "  Creating circle (radium: %d) :" % r
c = example.Circle(r)

# Set the location of the object

c.x = 20
c.y = 30
print "  Here is its current position:"
print "    Circle = (%f, %f)" % (c.x,c.y)

# ----- Call some methods -----

print "\n  Here are some properties of the Circle:"
print "    area      = ", c.area()
print "    perimeter = ", c.perimeter()
dx = 1;
dy = 1;
print "    Moving with (%d, %d)..." % (dx, dy)
c.move(dx, dy)

del c

print "==================================="

# test area function */
r = 1;
print "  Creating circle (radium: %d) :" % r 
c = example.Circle(r)
# Set the location of the object

c.x = 20
c.y = 30
print "  Here is its current position:"
print "    Circle = (%f, %f)" % (c.x,c.y)

# ----- Call some methods -----

print "\n  Here are some properties of the Circle:"
print "    area      = ", c.area()
