# Operator overloading example
require 'example'

a = Example::IntSum.new(0)
b = Example::DoubleSum.new(100.0)

# Use the objects.  They should be callable just like a normal
# Ruby function.

(0..100).each do |i|
  a.call(i)              # note: function call
  b.call(Math.sqrt(i))   # note: function call
end

puts a.result
puts b.result

