#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'dynamic_cast'

f = Dynamic_cast::Foo.new
b = Dynamic_cast::Bar.new

x = f.blah
y = b.blah

a = Dynamic_cast.do_test(y)
if a != "Bar::test"
  puts "Failed!!"
end
    
