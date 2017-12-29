package main

import "./extend_template"

func main() {
	f := extend_template.NewFoo_0()
	if f.Test1(37) != 37 {
		panic(0)
	}

	if f.Test2(42) != 42 {
		panic(0)
	}
}
