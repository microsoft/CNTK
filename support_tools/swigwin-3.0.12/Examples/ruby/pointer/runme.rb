# file: runme.rb

require 'example'

# First create some objects using the pointer library.
print "Testing the pointer library\n"
a = Example::new_intp()
b = Example::new_intp()
c = Example::new_intp()

Example::intp_assign(a,37)
Example::intp_assign(b,42)

print "     a = #{a}\n"
print "     b = #{b}\n"
print "     c = #{c}\n"

# Call the add() function with some pointers
Example::add(a, b, c)

# Now get the result
r = Example::intp_value(c)
print "     37 + 42 = #{r}\n"

# Clean up the pointers
Example::delete_intp(a)
Example::delete_intp(b)
Example::delete_intp(c)

# Now try the typemap library
# This should be much easier. Now how it is no longer
# necessary to manufacture pointers.

print "Trying the typemap library\n"
r = Example::sub(37, 42)
print "     37 - 42 = #{r}\n"

# Now try the version with multiple return values

print "Testing multiple return values\n"
q, r = Example::divide(42, 37)
print "     42/37 = #{q} remainder #{r}\n"



