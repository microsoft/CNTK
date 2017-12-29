import example 

# Call the Circle() function correctly

x = 1;
y = 1;
r = 3;

c = example.Circle(x, y, r)

print "The return value of Circle(%d, %d, %d) is %d" % (x,y,r,c)

# test pre-assertion
x = 1;
y = -1;
r = 3;

c = example.Circle(x, y, r)

print "The return value of Circle(%d, %d, %d) is %d" % (x,y,r,c)
