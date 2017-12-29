package main

import "./typedef_scope"

func main() {
	b := typedef_scope.NewBar()
	x := b.Test1(42, "hello")
	if x != 42 {
		panic("Failed!!")
	}

	xb := b.Test2(42, "hello")
	if xb != "hello" {
		panic("Failed!!")
	}
}
