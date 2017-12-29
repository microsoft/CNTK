# file: runme.rb

require 'example'

# Create a couple of a vectors

v = Example::new_Vector(1, 2, 3)
w = Example::new_Vector(10, 11, 12)

print "I just created the following vectors\n"
Example::vector_print(v)
Example::vector_print(w)

# Now call some of our functions

print "\nNow I'm going to compute the dot product\n"
d = Example::dot_product(v,w)
print "dot product = #{d} (should be 68)\n"

# Add the vectors together

print "\nNow I'm going to add the vectors together\n"
r = Example::vector_add(v,w)
Example::vector_print(r)
print "The value should be (11, 13, 15)\n"

# Now I'd better clean up the return result r

print "\nNow I'm going to clean up the return result\n"
Example::free(r)

print "Good\n"
