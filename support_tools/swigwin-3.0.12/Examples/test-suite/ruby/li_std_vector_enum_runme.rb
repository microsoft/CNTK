#!/usr/bin/env ruby

require 'swig_assert'
require 'li_std_vector_enum'
include Li_std_vector_enum

ev = EnumVector.new()

swig_assert(ev.nums[0] == 10)
swig_assert(ev.nums[1] == 20)
swig_assert(ev.nums[2] == 30)

it = ev.nums.begin
v = it.value()
swig_assert(v == 10)
it.next()
v = it.value()
swig_assert(v == 20)

expected = 10 
ev.nums.each do|val|
  swig_assert(val == expected)
  expected += 10
end

