package main

import "./ret_by_value"

func main() {
	a := ret_by_value.Get_test()
	if a.GetMyInt() != 100 {
		panic(0)
	}

	if a.GetMyShort() != 200 {
		panic(0)
	}
}
