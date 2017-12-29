#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'using_protected'

include Using_protected

f = FooBar.new
f.x = 3

if f.blah(4) != 4
  raise RuntimeError, "blah(int)"
end

