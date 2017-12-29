package main

import . "./cpp_static"

func main() {
	StaticFunctionTestStatic_func()
	StaticFunctionTestStatic_func_2(1)
	StaticFunctionTestStatic_func_3(1, 2)
	SetStaticMemberTestStatic_int(10)
	if GetStaticMemberTestStatic_int() != 10 {
		panic(GetStaticMemberTestStatic_int())
	}
}
