#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'using_inherit'

include Using_inherit

b = Bar.new
if b.test(3) != 3
  raise RuntimeError,"Bar::test(int)"
end

if b.test(3.5) != 3.5
  raise RuntimeError, "Bar::test(double)"
end

b = Bar2.new
if b.test(3) != 6
  raise RuntimeError,"Bar2::test(int)"
end

if b.test(3.5) != 7.0
  raise RuntimeError, "Bar2::test(double)"
end


b = Bar3.new
if b.test(3) != 6
  raise RuntimeError,"Bar3::test(int)"
end

if b.test(3.5) != 7.0
  raise RuntimeError, "Bar3::test(double)"
end


b = Bar4.new
if b.test(3) != 6
  raise RuntimeError,"Bar4::test(int)"
end

if b.test(3.5) != 7.0
  raise RuntimeError, "Bar4::test(double)"
end


b = Fred1.new
if b.test(3) != 3
  raise RuntimeError,"Fred1::test(int)"
end

if b.test(3.5) != 7.0
  raise RuntimeError, "Fred1::test(double)"
end


b = Fred2.new
if b.test(3) != 3
  raise RuntimeError,"Fred2::test(int)"
end

if b.test(3.5) != 7.0
  raise RuntimeError, "Fred2::test(double)"
end


