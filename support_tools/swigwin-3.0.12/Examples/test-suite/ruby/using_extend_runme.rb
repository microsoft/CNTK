#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'using_extend'

include Using_extend

f = FooBar.new
if f.blah(3) != 3
  raise RuntimeError,"blah(int)"
end

if f.blah(3.5) != 3.5
  raise RuntimeError,"blah(double)"
end

if f.blah("hello") != "hello"
  raise RuntimeError,"blah(char *)"
end

if f.blah(3,4) != 7
  raise RuntimeError,"blah(int,int)"
end

if f.blah(3.5,7.5) != (3.5+7.5)
  raise RuntimeError,"blah(double,double)"
end

