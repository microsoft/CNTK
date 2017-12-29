# Operator overloading example
require 'example'

include Example

a = Example::Complex.new(2, 3)
b = Example::Complex.new(-5, 10)

puts "a   = #{a}"
puts "b   = #{b}"

c = a + b
puts "c   = #{c}"
puts "a*b = #{a*b}"
puts "a-c = #{a-c}"

# This should invoke Complex's copy constructor
e = Example::Complex.new(a-c)
e = a - c
puts "e   = #{e}"

# Big expression
f = ((a+b)*(c+b*e)) + (-a)
puts "f   = #{f}"

