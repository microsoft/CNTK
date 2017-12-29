package main

import "./varargs_overload"

func main() {
	if varargs_overload.Vararg_over1("Hello") != "Hello" {
		panic(0)
	}
	if varargs_overload.Vararg_over1(2) != "2" {
		panic(1)
	}

	if varargs_overload.Vararg_over2("Hello") != "Hello" {
		panic(2)
	}
	if varargs_overload.Vararg_over2(2, 2.2) != "2 2.2" {
		panic(3)
	}

	if varargs_overload.Vararg_over3("Hello") != "Hello" {
		panic(4)
	}
	if varargs_overload.Vararg_over3(2, 2.2, "hey") != "2 2.2 hey" {
		panic(5)
	}

	if varargs_overload.Vararg_over4("Hello") != "Hello" {
		panic(6)
	}

	if varargs_overload.Vararg_over4(123) != "123" {
		panic(7)
	}

	if varargs_overload.Vararg_over4("Hello", 123) != "Hello" {
		panic(8)
	}
}
