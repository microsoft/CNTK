package main

import . "./extend_template_ns"

func main() {
	f := NewFoo_One()
	if f.Test1(37) != 37 {
		panic(0)
	}

	if f.Test2(42) != 42 {
		panic(0)
	}
}
