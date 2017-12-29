package main

import . "./disown"

func main() {
	a := NewA()

	b := NewB()
	b.Acquire(a)
}
