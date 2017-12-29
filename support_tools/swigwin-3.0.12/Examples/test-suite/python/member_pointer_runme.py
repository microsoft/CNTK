# Example using pointers to member functions

from member_pointer import *


def check(what, expected, actual):
    if expected != actual:
        raise RuntimeError(
            "Failed: ", what, " Expected: ", expected, " Actual: ", actual)

# Get the pointers

area_pt = areapt()
perim_pt = perimeterpt()

# Create some objects

s = Square(10)

# Do some calculations

check("Square area ", 100.0, do_op(s, area_pt))
check("Square perim", 40.0, do_op(s, perim_pt))

memberPtr = cvar.areavar
memberPtr = cvar.perimetervar

# Try the variables
check("Square area ", 100.0, do_op(s, cvar.areavar))
check("Square perim", 40.0, do_op(s, cvar.perimetervar))

# Modify one of the variables
cvar.areavar = perim_pt

check("Square perimeter", 40.0, do_op(s, cvar.areavar))

# Try the constants

memberPtr = AREAPT
memberPtr = PERIMPT
memberPtr = NULLPT

check("Square area ", 100.0, do_op(s, AREAPT))
check("Square perim", 40.0, do_op(s, PERIMPT))

check("Add by value", 3, call1(ADD_BY_VALUE, 1, 2))
check("Add by pointer", 7, call2(ADD_BY_POINTER, 3, 4))
check("Add by reference", 11, call3(ADD_BY_REFERENCE, 5, 6))
