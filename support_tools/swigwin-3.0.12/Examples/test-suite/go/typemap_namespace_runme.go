package main

import . "./typemap_namespace"

func main() {
	if Test1("hello") != "hello" {
		panic(0)
	}

	if Test2("hello") != "hello" {
		panic(0)
	}
}
