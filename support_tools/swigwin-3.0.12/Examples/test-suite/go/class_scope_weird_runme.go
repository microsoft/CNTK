package main

import "./class_scope_weird"

func main() {
	f := class_scope_weird.NewFoo()
	class_scope_weird.NewFoo(3)
	if f.Bar(3) != 3 {
		panic(f.Bar(3))
	}
}
