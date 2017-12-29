#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'struct_value'

b = Struct_value::Bar.new

b.a.x = 3
raise RuntimeError if b.a.x != 3

b.b.x = 3
raise RuntimeError if b.b.x != 3

