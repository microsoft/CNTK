package main

import "./special_variable_macros"

func main() {
	name := special_variable_macros.NewName()
	if special_variable_macros.TestFred(name) != "none" {
		panic("test failed")
	}
	if special_variable_macros.TestJack(name) != "$specialname" {
		panic("test failed")
	}
	if special_variable_macros.TestJill(name) != "jilly" {
		panic("test failed")
	}
	if special_variable_macros.TestMary(name) != "SWIGTYPE_p_NameWrap" {
		panic("test failed")
	}
	if special_variable_macros.TestJames(name) != "SWIGTYPE_Name" {
		panic("test failed")
	}
	if special_variable_macros.TestJim(name) != "multiname num" {
		panic("test failed")
	}
	if special_variable_macros.TestJohn(special_variable_macros.NewPairIntBool(10, false)) != 123 {
		panic("test failed")
	}
}
