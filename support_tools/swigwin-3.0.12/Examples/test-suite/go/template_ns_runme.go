package main

import . "./template_ns"

func main() {
	p1 := NewPairii(2, 3)
	p2 := NewPairii(p1)

	if p2.GetFirst() != 2 {
		panic(0)
	}
	if p2.GetSecond() != 3 {
		panic(0)
	}

	p3 := NewPairdd(3.5, 2.5)
	p4 := NewPairdd(p3)

	if p4.GetFirst() != 3.5 {
		panic(0)
	}

	if p4.GetSecond() != 2.5 {
		panic(0)
	}
}
