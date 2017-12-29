#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'extend_template'

f = Extend_template::Foo_0.new
if f.test1(37) != 37
  raise RuntimeError
end

if f.test2(42) != 42
  raise RuntimeError
end
