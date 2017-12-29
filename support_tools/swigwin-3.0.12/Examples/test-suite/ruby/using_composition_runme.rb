#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'using_composition'

include Using_composition

f = FooBar.new
if f.blah(3) != 3
  raise RuntimeError,"FooBar::blah(int)"
end

if f.blah(3.5) != 3.5
  raise RuntimeError,"FooBar::blah(double)"
end

if f.blah("hello") != "hello"
  raise RuntimeError,"FooBar::blah(char *)"
end


f = FooBar2.new
if f.blah(3) != 3
  raise RuntimeError,"FooBar2::blah(int)"
end

if f.blah(3.5) != 3.5
  raise RuntimeError,"FooBar2::blah(double)"
end

if f.blah("hello") != "hello"
  raise RuntimeError,"FooBar2::blah(char *)"
end


f = FooBar3.new
if f.blah(3) != 3
  raise RuntimeError,"FooBar3::blah(int)"
end

if f.blah(3.5) != 3.5
  raise RuntimeError,"FooBar3::blah(double)"
end

if f.blah("hello") != "hello"
  raise RuntimeError,"FooBar3::blah(char *)"
end

