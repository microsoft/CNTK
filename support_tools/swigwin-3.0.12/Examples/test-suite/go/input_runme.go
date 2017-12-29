package main

import . "./input"

func main() {
	f := NewFoo()
	if f.Foo(2) != 4 {
		panic(0)
	}

	if Sfoo("Hello") != "Hello world" {
		panic(0)
	}
}
