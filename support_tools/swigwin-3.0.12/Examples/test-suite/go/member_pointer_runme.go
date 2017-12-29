// Example using pointers to member functions

package main

import "fmt"
import . "./member_pointer"

func check(what string, expected float64, actual float64) {
	if expected != actual {
		panic(fmt.Sprintf("Failed: %s Expected: %f Actual; %f", what, expected, actual))
	}
}

func main() {
	// Get the pointers

	area_pt := Areapt()
	perim_pt := Perimeterpt()

	// Create some objects

	s := NewSquare(10)

	// Do some calculations

	check("Square area ", 100.0, Do_op(s, area_pt))
	check("Square perim", 40.0, Do_op(s, perim_pt))

	_ = GetAreavar()
	_ = GetPerimetervar()

	// Try the variables
	check("Square area ", 100.0, Do_op(s, GetAreavar()))
	check("Square perim", 40.0, Do_op(s, GetPerimetervar()))

	// Modify one of the variables
	SetAreavar(perim_pt)

	check("Square perimeter", 40.0, Do_op(s, GetAreavar()))

	// Try the constants

	_ = AREAPT
	_ = PERIMPT
	_ = NULLPT

	check("Square area ", 100.0, Do_op(s, AREAPT))
	check("Square perim", 40.0, Do_op(s, PERIMPT))
}
