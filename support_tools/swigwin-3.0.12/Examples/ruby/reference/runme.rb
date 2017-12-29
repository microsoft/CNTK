# file: runme.rb

# This file illustrates the manipulation of C++ references in Ruby.

require 'example'

# ----- Object creation -----

print "Creating some objects:\n"
a = Example::Vector.new(3,4,5)
b = Example::Vector.new(10,11,12)

print "    Created ", a.print, "\n"
print "    Created ", b.print, "\n"

# ----- Call an overloaded operator -----

# This calls the wrapper we placed around
#
#      operator+(const Vector &a, const Vector &) 
#
# It returns a new allocated object.

print "Adding a+b\n"
c = Example::addv(a, b)
print "    a+b = ", c.print, "\n"

# ----- Create a vector array -----

print "Creating an array of vectors\n"
va = Example::VectorArray.new(10)
print "    va = #{va}\n"

# ----- Set some values in the array -----

# These operators copy the value of a and b to the vector array
va.set(0, a)
va.set(1, b)

va.set(2, Example::addv(a,b))

c = Example::addv(a,b)
va.set(3, c)

=begin commented out due to GC issue

# Get some values from the array

print "Getting some array values\n"
for i in 0...5
  print "    va(#{i}) = ", va.get(i).print, "\n"
end

# Watch under resource meter to check on this
print "Making sure we don't leak memory.\n"
for i in 0...1000000
  c = va.get(i % 10)
end

=end
