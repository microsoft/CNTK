package main

import "./varargs"

func main() {
	if varargs.Test("Hello") != "Hello" {
		panic("Failed")
	}

	f := varargs.NewFoo("Greetings")
	if f.GetStr() != "Greetings" {
		panic("Failed")
	}

	if f.Test("Hello") != "Hello" {
		panic("Failed")
	}

	if varargs.Test_def("Hello", 1) != "Hello" {
		panic("Failed")
	}

	if varargs.Test_def("Hello") != "Hello" {
		panic("Failed")
	}
}
