package main

import . "./extend_variable"

func main() {
	if FooBar != 42 {
		panic(0)
	}
}
