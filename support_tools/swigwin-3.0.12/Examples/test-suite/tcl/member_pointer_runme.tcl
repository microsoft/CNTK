# Example using pointers to member functions

if [ catch { load ./member_pointer[info sharedlibextension] member_pointer} err_msg ] {
	puts stderr "Could not load shared object:\n$err_msg"
}

proc check {what expected actual} {
  if {$expected != $actual } {
    error "Failed: $what , Expected: $expected , Actual: $actual"
  }
}
# Get the pointers

set area_pt [ areapt ]
set perim_pt [ perimeterpt ]

# Create some objects

set s [Square -args 10]

# Do some calculations

check "Square area " 100.0 [do_op $s $area_pt]
check "Square perim" 40.0 [do_op $s $perim_pt]

set memberPtr $areavar
set memberPtr $perimetervar

# Try the variables
check "Square area " 100.0 [do_op $s $areavar]
check "Square perim" 40.0 [do_op $s $perimetervar]

# Modify one of the variables
set areavar $perim_pt

check "Square perimeter" 40.0 [do_op $s $areavar]

# Try the constants

set memberPtr $AREAPT
set memberPtr $PERIMPT
set memberPtr $NULLPT

check "Square area " 100.0 [do_op $s $AREAPT]
check "Square perim" 40.0 [do_op $s $PERIMPT]

