package main

import . "./li_carrays"

func main() {
	d := NewDoubleArray(10)

	d.Setitem(0, 7)
	d.Setitem(5, d.Getitem(0)+3)

	if d.Getitem(5)+d.Getitem(0) != 17 {
		panic(0)
	}
}
