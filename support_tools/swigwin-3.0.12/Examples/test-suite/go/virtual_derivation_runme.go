package main

import . "./virtual_derivation"

// very innocent example

func main() {
	b := NewB(3)
	if b.Get_a() != b.Get_b() {
		panic("something is really wrong")
	}
}
