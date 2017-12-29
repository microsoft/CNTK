package main

import "./template_extend1"

func main() {
	a := template_extend1.NewLBaz()
	b := template_extend1.NewDBaz()

	if a.Foo() != "lBaz::foo" {
		panic(0)
	}

	if b.Foo() != "dBaz::foo" {
		panic(0)
	}
}
