#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'smart_pointer_not'

include Smart_pointer_not

f = Foo.new
b = Bar.new(f)
s = Spam.new(f)
g = Grok.new(f)

begin
  x = b.x
  puts "Error! b.x"
rescue NameError
end

begin
  x = s.x
  puts "Error! s.x"    
rescue NameError
end

begin
  x = g.x
  puts "Error! g.x"
rescue NameError
end

begin
  x = b.getx()
  puts "Error! b.getx()"    
rescue NameError
end

begin
  x = s.getx()
  puts "Error! s.getx()"        
rescue NameError
end

begin
  x = g.getx()
  puts "Error! g.getx()"
rescue NameError
end
