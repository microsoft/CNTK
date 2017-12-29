#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'extend_template_ns'

include Extend_template_ns

f = Foo_One.new
if f.test1(37) != 37
  raise RuntimeError
end

if f.test2(42) != 42
  raise RuntimeError
end
