package main

import . "./smart_pointer_rename"

func main() {
	f := NewFoo()
	b := NewBar(f)

	if b.Test() != 3 {
		panic(0)
	}

	if b.Ftest1(1) != 1 {
		panic(0)
	}

	if b.Ftest2(2, 3) != 2 {
		panic(0)
	}
}
