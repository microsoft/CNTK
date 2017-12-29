package main

import . "./using_protected"

func main() {
	f := NewFooBar()
	f.SetX(3)

	if f.Blah(4) != 4 {
		panic("blah(int)")
	}
}
