package main

import "./friends"

func main() {
	a := friends.NewA(2)

	if friends.Get_val1(a).(int) != 2 {
		panic(0)
	}
	if friends.Get_val2(a) != 4 {
		panic(0)
	}
	if friends.Get_val3(a) != 6 {
		panic(0)
	}

	// nice overload working fine
	if friends.Get_val1(1, 2, 3).(int) != 1 {
		panic(0)
	}

	b := friends.NewB(3)

	// David's case
	if friends.Mix(a, b) != 5 {
		panic(0)
	}

	di := friends.NewD_d(2)
	dd := friends.NewD_d(3.3)

	// incredible template overloading working just fine
	if friends.Get_val1(di).(float64) != 2 {
		panic(0)
	}
	if friends.Get_val1(dd).(float64) != 3.3 {
		panic(0)
	}

	friends.Set(di, 4.0)
	friends.Set(dd, 1.3)

	if friends.Get_val1(di).(float64) != 4 {
		panic(0)
	}
	if friends.Get_val1(dd).(float64) != 1.3 {
		panic(0)
	}
}
