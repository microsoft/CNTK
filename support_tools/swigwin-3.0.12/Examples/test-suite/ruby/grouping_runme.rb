#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'grouping'

x = Grouping.test1(42)
if x != 42
  raise RuntimeError
end

Grouping.test2(42)

x = Grouping.do_unary(37, Grouping::NEGATE)
if x != -37
  raise RuntimeError
end

Grouping.test3 = 42
