package main

import . "./smart_pointer_multi"

func main() {
	f := NewFoo()
	b := NewBar(f)
	s := NewSpam(b)
	g := NewGrok(b)

	s.SetX(3)
	if s.Getx() != 3 {
		panic(0)
	}

	g.SetX(4)
	if g.Getx() != 4 {
		panic(0)
	}
}
