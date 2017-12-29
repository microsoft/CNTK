package main

import "./dynamic_cast"

func main() {
	f := dynamic_cast.NewFoo()
	b := dynamic_cast.NewBar()

	_ = f.Blah()
	y := b.Blah()

	a := dynamic_cast.Do_test(dynamic_cast.FooToBar(y))
	if a != "Bar::test" {
		panic("Failed!!")
	}
}
