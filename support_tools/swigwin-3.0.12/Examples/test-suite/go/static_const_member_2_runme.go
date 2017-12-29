package main

import . "./static_const_member_2"

func main() {
	_ = NewTest_int()

	_ = CavityPackFlagsForward_field
	_ = Test_intCurrent_profile
	_ = Test_intRightIndex
	_ = CavityPackFlagsBackward_field
	_ = Test_intLeftIndex
	// _ = GetTest_int_Cavity_flags()

	if GetFooBAZ().GetVal() != 2*GetFooBAR().GetVal() {
		panic(0)
	}
}
