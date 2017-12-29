package main

import . "./array_member"

func main() {
	f := NewFoo()
	f.SetData(GetGlobal_data())

	for i := 0; i < 8; i++ {
		if Get_value(f.GetData(), i) != Get_value(GetGlobal_data(), i) {
			panic("Bad array assignment")
		}
	}

	for i := 0; i < 8; i++ {
		Set_value(f.GetData(), i, -i)
	}

	SetGlobal_data(f.GetData())

	for i := 0; i < 8; i++ {
		if Get_value(f.GetData(), i) != Get_value(GetGlobal_data(), i) {
			panic("Bad array assignment")
		}
	}
}
