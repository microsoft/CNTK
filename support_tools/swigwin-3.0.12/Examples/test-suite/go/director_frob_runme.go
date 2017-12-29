package main

import . "./director_frob"

func main() {
	foo := NewBravo()
	s := foo.Abs_method()

	if s != "Bravo::abs_method()" {
		panic(s)
	}
}
