package main

import "./overload_extend2"

func main() {
	f := overload_extend2.NewFoo()
	if f.Test(3) != 1 {
		panic(0)
	}
	if f.Test("hello") != 2 {
		panic(0)
	}
	if f.Test(3.5, 2.5) != 3 {
		panic(0)
	}
	if f.Test("hello", 20) != 1020 {
		panic(0)
	}
	if f.Test("hello", 20, 100) != 120 {
		panic(0)
	}

	// C default args
	if f.Test(f) != 30 {
		panic(0)
	}
	if f.Test(f, 100) != 120 {
		panic(0)
	}
	if f.Test(f, 100, 200) != 300 {
		panic(0)
	}
}
