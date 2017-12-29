package main

import . "./overload_complicated"

func main() {
	var pInt *int

	// Check the correct constructors are available
	p := NewPop(pInt)

	p = NewPop(pInt, false)

	// Check overloaded in const only and pointers/references
	// which target languages cannot disambiguate
	if p.Hip(false) != 701 {
		panic("Test 1 failed")
	}

	if p.Hip(pInt) != 702 {
		panic("Test 2 failed")
	}

	// Reverse the order for the above
	if p.Hop(pInt) != 805 {
		panic("Test 3 failed")
	}

	if p.Hop(false) != 801 {
		panic("Test 4 failed")
	}

	// Few more variations and order shuffled
	if p.Pop(false) != 901 {
		panic("Test 5 failed")
	}

	if p.Pop(pInt) != 902 {
		panic("Test 6 failed")
	}

	if p.Pop() != 905 {
		panic("Test 7 failed")
	}

	// Overload on const only
	if p.Bop(pInt) != 1001 {
		panic("Test 8 failed")
	}

	if p.Bip(pInt) != 2001 {
		panic("Test 9 failed")
	}

	// Globals
	if Muzak(false) != 3001 {
		panic("Test 10 failed")
	}

	if Muzak(pInt) != 3002 {
		panic("Test 11 failed")
	}
}
