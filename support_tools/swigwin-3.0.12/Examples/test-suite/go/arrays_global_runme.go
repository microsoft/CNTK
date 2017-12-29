package main

import . "./arrays_global"

func main() {
	SetArray_i(GetArray_const_i())

	GetBeginString_FIX44a()
	GetBeginString_FIX44b()
	GetBeginString_FIX44c()
	GetBeginString_FIX44d()
	GetBeginString_FIX44d()
	SetBeginString_FIX44b("12\00045")
	GetBeginString_FIX44b()
	GetBeginString_FIX44d()
	GetBeginString_FIX44e()
	GetBeginString_FIX44f()

	Test_a("hello", "hi", "chello", "chi")

	Test_b("1234567", "hi")
}
