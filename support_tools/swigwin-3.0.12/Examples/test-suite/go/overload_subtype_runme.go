package main

import . "./overload_subtype"

func main() {
	f := NewFoo()
	b := NewBar()

	if Spam(f) != 1 {
		panic("foo")
	}

	if Spam(b) != 2 {
		panic("bar")
	}
}
