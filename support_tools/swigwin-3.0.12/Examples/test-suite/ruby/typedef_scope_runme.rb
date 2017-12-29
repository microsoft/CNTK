#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'typedef_scope'

b = Typedef_scope::Bar.new

x = b.test1(42, "hello")
if x != 42
  puts "Failed!!"
end

x = b.test2(42, "hello")
if x != "hello"
  puts "Failed!!"
end
