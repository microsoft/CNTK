package main

import . "./compactdefaultargs"

func main() {
	defaults1 := NewDefaults1(1000)
	defaults1 = NewDefaults1()

	if defaults1.Ret(10.0) != 10.0 {
		println(1, defaults1.Ret(10.0))
		panic(defaults1.Ret(10.0))
	}

	if defaults1.Ret() != -1.0 {
		println(2, defaults1.Ret())
		panic(defaults1.Ret())
	}

	defaults2 := NewDefaults2(1000)
	defaults2 = NewDefaults2()

	if defaults2.Ret(10.0) != 10.0 {
		panic(defaults2.Ret(10.0))
	}

	if defaults2.Ret() != -1.0 {
		panic(defaults2.Ret())
	}
}
