package main

import . "./using_private"

func main() {
	f := NewFooBar()
	f.SetX(3)

	if f.Blah(4) != 4 {
		panic("blah(int)")
	}

	if f.Defaulted() != -1 {
		panic("defaulted()")
	}

	if f.Defaulted(222) != 222 {
		panic("defaulted(222)")
	}
}
