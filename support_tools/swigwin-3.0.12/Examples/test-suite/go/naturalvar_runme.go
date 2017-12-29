package main

import . "./naturalvar"

func main() {
	f := NewFoo()
	b := NewBar()

	b.SetF(f)

	SetS("hello")
	b.SetS("hello")

	if b.GetS() != GetS() {
		panic(0)
	}
}
