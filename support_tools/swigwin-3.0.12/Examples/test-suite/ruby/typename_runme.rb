#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'typename'

f = Typename::Foo.new
b = Typename::Bar.new

x = Typename.twoFoo(f)
unless x.is_a? Float
  puts "Wrong return type!"
end

y = Typename.twoBar(b)
unless y.is_a? Integer
  puts "Wrong return type!"
end
