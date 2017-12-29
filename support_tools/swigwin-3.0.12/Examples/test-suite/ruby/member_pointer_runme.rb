#!/usr/bin/env ruby
#
# Example using pointers to member functions
# 
# 
#

require 'swig_assert'

require 'member_pointer'

include Member_pointer

def check(what, expected, actual)
  if not expected == actual
    raise RuntimeError, "Failed: #{what} Expected: #{expected} Actual: #{actual}"
  end
end

# Get the pointers

area_pt = Member_pointer::areapt
perim_pt = Member_pointer::perimeterpt

# Create some objects

s = Member_pointer::Square.new(10)

# Do some calculations

check "Square area ", 100.0, Member_pointer::do_op(s, area_pt) 
check "Square perim", 40.0, Member_pointer::do_op(s, perim_pt)

memberPtr = Member_pointer::areavar
memberPtr = Member_pointer::perimetervar

# Try the variables
check "Square area ", 100.0, Member_pointer::do_op(s, Member_pointer::areavar)
check "Square perim", 40.0, Member_pointer::do_op(s, Member_pointer::perimetervar)

# Modify one of the variables
Member_pointer::areavar = perim_pt

check "Square perimeter", 40.0, Member_pointer::do_op(s, Member_pointer::areavar)

# Try the constants

memberPtr = Member_pointer::AREAPT
memberPtr = Member_pointer::PERIMPT
memberPtr = Member_pointer::NULLPT

check "Square area ", 100.0, Member_pointer::do_op(s, Member_pointer::AREAPT)
check "Square perim", 40.0, Member_pointer::do_op(s, Member_pointer::PERIMPT)

