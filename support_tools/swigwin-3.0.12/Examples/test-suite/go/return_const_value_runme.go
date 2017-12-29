package main

import "./return_const_value"

func main() {
	p := return_const_value.Foo_ptrGetPtr()
	if p.GetVal() != 17 {
		panic("Runtime test1 failed")
	}

	p = return_const_value.Foo_ptrGetConstPtr()
	if p.GetVal() != 17 {
		panic("Runtime test2 failed")
	}
}
