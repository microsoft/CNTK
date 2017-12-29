package main

import . "./smart_pointer_simple"

func main() {
	f := NewFoo()
	b := NewBar(f)

	b.SetX(3)
	if b.Getx() != 3 {
		panic(0)
	}

	fp := b.X__deref__()
	fp.SetX(4)
	if fp.Getx() != 4 {
		panic(0)
	}
}
