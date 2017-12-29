#!/usr/bin/env ruby
#
# Test run of arrays.i
#
#

require 'swig_assert'
require 'arrays'

include Arrays

a = SimpleStruct.new
a.double_field = 2.0

b = SimpleStruct.new
b.double_field = 1.0

# @bug:  this is broken
#
# c = [a,b]
# fn_taking_arrays(c)
#
# a = ArrayStruct.new
# a.array_i[0] = 0
#
