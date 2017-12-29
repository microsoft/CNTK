package main

import "./overload_extend_c"

func main() {
	f := overload_extend_c.NewFoo()
	if f.Test().(int) != 0 {
		panic(0)
	}
	if f.Test(3).(int) != 1 {
		panic(0)
	}
	if f.Test("hello").(int) != 2 {
		panic(0)
	}
	if f.Test(float64(3), float64(2)).(float64) != 5 {
		panic(0)
	}
	if f.Test(3.0).(float64) != 1003 {
		panic(0)
	}
}
