package main

import . "./smart_pointer_overload"

func main() {
	f := NewFoo()
	b := NewBar(f)

	if f.Test(3) != 1 {
		panic(0)
	}
	if f.Test(3.5) != 2 {
		panic(0)
	}
	if f.Test("hello") != 3 {
		panic(0)
	}

	if b.Test(3) != 1 {
		panic(0)
	}
	if b.Test(3.5) != 2 {
		panic(0)
	}
	if b.Test("hello") != 3 {
		panic(0)
	}
}
