# file: runme.rb
# Test various properties of classes defined in separate modules

puts "Testing the %import directive"

require 'base'
require 'foo'
require 'bar'
require 'spam'

# Create some objects

puts "Creating some objects"

a = Base::Base.new
b = Foo::Foo.new
c = Bar::Bar.new
d = Spam::Spam.new

# Try calling some methods
puts "Testing some methods"
puts "Should see 'Base::A' ---> #{a.A}"
puts "Should see 'Base::B' ---> #{a.B}"

puts  "Should see 'Foo::A' ---> #{b.A}"
puts  "Should see 'Foo::B' ---> #{b.B}"

puts  "Should see 'Bar::A' ---> #{c.A}"
puts  "Should see 'Bar::B' ---> #{c.B}"

puts  "Should see 'Spam::A' ---> #{d.A}"
puts  "Should see 'Spam::B' ---> #{d.B}"

# Try some casts

puts "\nTesting some casts\n"

x = a.toBase
puts "Should see 'Base::A' ---> #{x.A}"
puts "Should see 'Base::B' ---> #{x.B}"

x = b.toBase
puts "Should see 'Foo::A' ---> #{x.A}"
puts "Should see 'Base::B' ---> #{x.B}"

x = c.toBase
puts "Should see 'Bar::A' ---> #{x.A}"
puts "Should see 'Base::B' ---> #{x.B}"

x = d.toBase
puts "Should see 'Spam::A' ---> #{x.A}"
puts "Should see 'Base::B' ---> #{x.B}"

x = d.toBar
puts "Should see 'Bar::B' ---> #{x.B}"

puts "\nTesting some dynamic casts\n"
x = d.toBase

puts " Spam -> Base -> Foo : "
y = Foo::Foo.fromBase(x)
if y != nil
  puts "bad swig"
else
  puts "good swig"
end

puts " Spam -> Base -> Bar : "
y = Bar::Bar.fromBase(x)
if y != nil
  puts "good swig"
else
  puts "bad swig"
end
      
puts " Spam -> Base -> Spam : "
y = Spam::Spam.fromBase(x)
if y != nil
  puts "good swig"
else
  puts "bad swig"
end

puts " Foo -> Spam : "
y = Spam::Spam.fromBase(b)
if y != nil
  puts "bad swig"
else
  puts "good swig"
end
