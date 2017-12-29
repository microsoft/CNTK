#!/usr/bin/env ruby
#
# Put description here
#
# 
# 
# 
#

require 'swig_assert'

require 'li_carrays_cpp'

include Li_carrays_cpp

#
# Testing for %array_functions(int,intArray)
#
ary = new_intArray(2)
intArray_setitem(ary, 0, 0)
intArray_setitem(ary, 1, 1)
intArray_getitem(ary, 0)
intArray_getitem(ary, 1)
delete_intArray(ary)

#
# Testing for %array_class(double, doubleArray)
#
ary = DoubleArray.new(2)
ary[0] = 0.0
ary[1] = 1.0
ary[0]
ary[1]
ptr = ary.cast
ary2 = DoubleArray.frompointer(ptr)

