package main

import . "./rename_scope"

func main() {
	a := NewNatural_UP()
	b := NewNatural_BP()

	if a.Rtest() != 1 {
		panic(0)
	}

	if b.Rtest() != 1 {
		panic(0)
	}

	_ = Equals
}
