# file: runme.rb

require 'example'

# Call some templated functions
puts Example::maxint(3, 7)
puts Example::maxdouble(3.14, 2.18)

# Create some class

iv = Example::Vecint.new(100)
dv = Example::Vecdouble.new(1000)

100.times { |i| iv.setitem(i, 2*i) }

1000.times { |i| dv.setitem(i, 1.0/(i+1)) }

sum = 0
100.times { |i| sum = sum + iv.getitem(i) }

puts sum

sum = 0.0
1000.times { |i| sum = sum + dv.getitem(i) }
puts sum
